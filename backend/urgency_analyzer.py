import re
from dataclasses import dataclass, field
from typing import List, Optional

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from data_loader import ESCALATION_MAP

# ---------------------------------------------------------------------
# Keyword layer
# ---------------------------------------------------------------------

# Concrete, real-world risk factors — these are allowed to move the
# urgency score because they describe circumstances, not tone.
SUBSTANTIVE_RISK_KEYWORDS = [
    # medical / health dependency
    "insulin", "medication", "dialysis", "oxygen", "prescription",
    "refrigerated medicine", "medical condition", "medical appointment",
    # disability / accessibility
    "wheelchair", "disability", "disabled", "ramp", "mobility aid",
    "inaccessible", "accessible exit",
    # vulnerable people
    "infant", "baby", "newborn", "children", "child", "elderly",
    "pregnant",
    # essential services loss
    "no power", "no electricity", "disconnected", "no water",
    "cut off", "no gas", "no heating", "lift has broken", "lift is broken",
    # homelessness / housing crisis
    "evicted", "eviction", "nowhere to sleep", "sleeping in my car",
    "homeless",
    # safety / threat
    "threat", "threatening", "scared", "violence", "unsafe", "danger",
    "hurt me", "hurt myself",
    # legal / time-sensitive
    "legal deadline", "court date", "deadline", "expires",
    # financial hardship affecting essentials
    "no food", "no money left", "can't afford", "hardship",
]

# Surface-level urgency language — informational only, NEVER scored.
# This is exactly what fooled a naive system on the "URGENT!!! password"
# trap case in the training data.
ATTENTION_MARKER_WORDS = ["urgent", "asap", "emergency", "immediately", "right away"]


def _keyword_signal(message: str) -> tuple[float, List[str], bool]:
    """
    Returns:
      - a 0-1 score from SUBSTANTIVE risk keywords only
      - the list of matched substantive keywords (for explanation)
      - whether surface-level "attention language" was detected
        (reported to the human, but not scored)
    """
    text = message.lower()
    matches = [kw for kw in SUBSTANTIVE_RISK_KEYWORDS if kw in text]
    score = min(1.0, len(matches) / 3.0)

    attention_detected = (
        any(w in text for w in ATTENTION_MARKER_WORDS)
        or message.count("!") >= 2
        or bool(re.search(r"\b[A-Z]{4,}\b", message))  # shouty ALL-CAPS word
    )

    return score, matches, attention_detected


# ---------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------

@dataclass
class AnalysisResult:
    message: str
    urgency: str                       # "Critical" / "High" / "Normal"
    urgency_confidence: float          # 0.0 - 1.0
    category: str
    category_confidence: float
    route: str
    escalation: str                    # derived from urgency, e.g. "Yes – Immediate"
    escalate_to_human: bool
    matched_risk_keywords: List[str] = field(default_factory=list)
    attention_language_detected: bool = False
    explanation: str = ""

    def to_dict(self) -> dict:
        return {
            "message": self.message,
            "urgency": self.urgency,
            "urgency_confidence": round(self.urgency_confidence, 3),
            "category": self.category,
            "category_confidence": round(self.category_confidence, 3),
            "route": self.route,
            "escalation": self.escalation,
            "escalate_to_human": self.escalate_to_human,
            "matched_risk_keywords": self.matched_risk_keywords,
            "attention_language_detected": self.attention_language_detected,
            "explanation": self.explanation,
        }


# Below this confidence, flag for human review even if a class was predicted.
#
# NOTE ON THIS NUMBER: with only 30 training examples and 3 balanced
# classes, this classifier's confidence scores sit mostly in the
# 0.41-0.63 range even on cases it gets right (see evaluate.py output)
# — there just isn't enough data yet to produce sharply separated
# probabilities. 0.45 was chosen by inspecting that distribution: it
# catches the genuinely near-chance-level calls without escalating
# every single message (which a stricter threshold like 0.55 would do
# and would defeat the point of confidence-based triage). Re-run
# evaluate.py and re-tune this whenever the training set grows.
ESCALATION_CONFIDENCE_THRESHOLD = 0.45


def _make_pipeline() -> Pipeline:
    """
    Small, fast text classifier. With ~30 labeled examples this will
    not generalise perfectly — that's expected at this stage. It should
    be retrained as soon as more labeled cases are available (see
    /train endpoint and evaluate.py for an honest accuracy estimate).
    """
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, stop_words="english")),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])


class UrgencyAnalyzer:
    """
    Trains and serves three classifiers (urgency, category, route) off
    the same labeled CSV, plus the guarded keyword layer above.
    """

    def __init__(self):
        self.urgency_pipeline: Optional[Pipeline] = None
        self.category_pipeline: Optional[Pipeline] = None
        self.route_pipeline: Optional[Pipeline] = None
        self.is_trained = False

    # ---- Training -----------------------------------------------------

    def train(self, df: pd.DataFrame) -> dict:
        """
        `df` must have 'message', 'urgency', 'category', 'route' columns
        (see data_loader.load_training_data). Category/route classifiers
        are only trained if that column has >=2 distinct values in the
        data; otherwise those predictions fall back to "Unclassified".
        """
        info = {"n_examples": len(df)}

        if df["urgency"].nunique() < 2:
            self.urgency_pipeline = None
            info["urgency_trained"] = False
            info["urgency_reason"] = "Only one urgency class present in training data."
        else:
            self.urgency_pipeline = _make_pipeline()
            self.urgency_pipeline.fit(df["message"], df["urgency"])
            info["urgency_trained"] = True
            info["urgency_classes"] = sorted(df["urgency"].unique().tolist())

        if df["category"].nunique() >= 2:
            self.category_pipeline = _make_pipeline()
            self.category_pipeline.fit(df["message"], df["category"])
            info["category_trained"] = True
            info["category_classes"] = sorted(df["category"].unique().tolist())
        else:
            self.category_pipeline = None
            info["category_trained"] = False

        if df["route"].nunique() >= 2:
            self.route_pipeline = _make_pipeline()
            self.route_pipeline.fit(df["message"], df["route"])
            info["route_trained"] = True
        else:
            self.route_pipeline = None
            info["route_trained"] = False

        self.is_trained = self.urgency_pipeline is not None
        return info

    # ---- Inference ------------------------------------------------

    @staticmethod
    def _predict_with_confidence(pipeline: Optional[Pipeline], message: str, fallback: str):
        if pipeline is None:
            return fallback, 0.0
        proba = pipeline.predict_proba([message])[0]
        classes = pipeline.classes_
        best_idx = proba.argmax()
        return classes[best_idx], float(proba[best_idx])

    def analyze(self, message: str) -> AnalysisResult:
        kw_score, kw_matches, attention_detected = _keyword_signal(message)

        if self.urgency_pipeline is not None:
            ml_label, ml_conf = self._predict_with_confidence(
                self.urgency_pipeline, message, fallback="Normal"
            )
            # Blend the ML confidence for the predicted class with the
            # keyword score, but ONLY as a small nudge — ML does the
            # heavy lifting since it can read context, unlike keywords.
            urgency_confidence = min(1.0, 0.8 * ml_conf + 0.2 * kw_score)
            urgency = ml_label
        else:
            # No trained model yet — fall back to keyword score only,
            # mapped onto the three tiers.
            urgency_confidence = kw_score
            if kw_score >= 0.66:
                urgency = "Critical"
            elif kw_score >= 0.33:
                urgency = "High"
            else:
                urgency = "Normal"

        category, category_confidence = self._predict_with_confidence(
            self.category_pipeline, message, fallback="Unclassified"
        )
        route, _ = self._predict_with_confidence(
            self.route_pipeline, message, fallback=category
        )

        escalation = ESCALATION_MAP.get(urgency, "No")

        # Escalate to a human whenever: it's Critical (high stakes,
        # always worth a human glance), OR the model isn't confident.
        escalate_to_human = (
            urgency == "Critical" or urgency_confidence < ESCALATION_CONFIDENCE_THRESHOLD
        )

        explanation_parts = []
        if kw_matches:
            explanation_parts.append(f"Risk factors identified: {', '.join(kw_matches)}.")
        if attention_detected:
            explanation_parts.append(
                "Uses urgent-sounding language, but this alone does not "
                "raise the urgency score — verify against risk factors above."
            )
        if self.urgency_pipeline is not None:
            explanation_parts.append(
                f"Model classified as '{urgency}' with {urgency_confidence:.0%} confidence."
            )
        else:
            explanation_parts.append(
                "Classifier not yet trained — using keyword-only estimate."
            )
        if escalate_to_human:
            explanation_parts.append("Flagged for human review before action.")

        return AnalysisResult(
            message=message,
            urgency=urgency,
            urgency_confidence=urgency_confidence,
            category=category,
            category_confidence=category_confidence,
            route=route,
            escalation=escalation,
            escalate_to_human=escalate_to_human,
            matched_risk_keywords=kw_matches,
            attention_language_detected=attention_detected,
            explanation=" ".join(explanation_parts),
        )

    def analyze_batch(self, messages: pd.Series) -> List[AnalysisResult]:
        return [self.analyze(m) for m in messages]

    @staticmethod
    def summarize(results: List[AnalysisResult]) -> dict:
        """Summary stats for the Streamlit dashboard."""
        n = len(results)
        if n == 0:
            return {
                "total_messages": 0,
                "by_urgency": {}, "by_category": {},
                "escalated_count": 0, "escalation_rate": 0.0,
                "average_urgency_confidence": 0.0,
            }

        by_urgency = pd.Series([r.urgency for r in results]).value_counts().to_dict()
        by_category = pd.Series([r.category for r in results]).value_counts().to_dict()
        escalated_count = sum(1 for r in results if r.escalate_to_human)
        avg_conf = sum(r.urgency_confidence for r in results) / n

        return {
            "total_messages": n,
            "by_urgency": by_urgency,
            "by_category": by_category,
            "escalated_count": escalated_count,
            "escalation_rate": round(escalated_count / n, 3),
            "average_urgency_confidence": round(avg_conf, 3),
        }