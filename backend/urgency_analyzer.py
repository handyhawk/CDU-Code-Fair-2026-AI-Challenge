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

# Concrete risk factors.
# These describe actual circumstances, not just urgent-sounding language.
SUBSTANTIVE_RISK_KEYWORDS = [
    # Medical / health dependency
    "insulin",
    "medication",
    "dialysis",
    "oxygen",
    "prescription",
    "refrigerated medicine",
    "medical condition",
    "medical appointment",

    # Disability / accessibility
    "wheelchair",
    "disability",
    "disabled",
    "ramp",
    "mobility aid",
    "inaccessible",
    "accessible exit",

    # Vulnerable people
    "infant",
    "baby",
    "newborn",
    "children",
    "child",
    "elderly",
    "pregnant",

    # Essential services loss
    "no power",
    "no electricity",
    "disconnected",
    "no water",
    "cut off",
    "no gas",
    "no heating",
    "lift has broken",
    "lift is broken",

    # Homelessness / housing crisis
    "evicted",
    "eviction",
    "nowhere to sleep",
    "sleeping in my car",
    "homeless",

    # Safety / threat
    "threat",
    "threatening",
    "scared",
    "violence",
    "unsafe",
    "danger",
    "hurt me",
    "hurt myself",

    # Legal / time-sensitive
    "legal deadline",
    "court date",
    "deadline",
    "expires",

    # Financial hardship affecting essentials
    "no food",
    "no money left",
    "can't afford",
    "hardship",
]


# Surface-level urgency language.
# These are reported but do not directly raise urgency.
ATTENTION_MARKER_WORDS = [
    "urgent",
    "asap",
    "emergency",
    "immediately",
    "right away",
]


def _keyword_signal(
    message: str,
) -> tuple[float, List[str], bool]:
    """
    Returns:

    1. A 0-1 score based on substantive risk keywords.
    2. The matched risk keywords.
    3. Whether attention-grabbing language was detected.
    """

    text = message.lower()

    matches = [
        keyword
        for keyword in SUBSTANTIVE_RISK_KEYWORDS
        if keyword in text
    ]

    # Three or more substantive risk matches reaches the maximum score.
    score = min(
        1.0,
        len(matches) / 3.0,
    )

    attention_detected = (
        any(
            word in text
            for word in ATTENTION_MARKER_WORDS
        )
        or message.count("!") >= 2
        or bool(
            re.search(
                r"\b[A-Z]{4,}\b",
                message,
            )
        )
    )

    return (
        score,
        matches,
        attention_detected,
    )


# ---------------------------------------------------------------------
# Result object
# ---------------------------------------------------------------------

@dataclass
class AnalysisResult:
    message: str

    urgency: str
    urgency_confidence: float

    category: str
    category_confidence: float

    route: str

    escalation: str
    escalate_to_human: bool

    matched_risk_keywords: List[str] = field(
        default_factory=list
    )

    attention_language_detected: bool = False

    explanation: str = ""

    def to_dict(self) -> dict:
        return {
            "message": self.message,

            "urgency": self.urgency,
            "urgency_confidence": round(
                self.urgency_confidence,
                3,
            ),

            "category": self.category,
            "category_confidence": round(
                self.category_confidence,
                3,
            ),

            "route": self.route,

            "escalation": self.escalation,

            "escalate_to_human":
                self.escalate_to_human,

            "matched_risk_keywords":
                self.matched_risk_keywords,

            "attention_language_detected":
                self.attention_language_detected,

            "explanation":
                self.explanation,
        }


# ---------------------------------------------------------------------
# Confidence threshold
# ---------------------------------------------------------------------

# With only a small training dataset, the local ML probabilities are
# relatively weak. Below this threshold, force human review.
ESCALATION_CONFIDENCE_THRESHOLD = 0.45


# ---------------------------------------------------------------------
# ML pipeline
# ---------------------------------------------------------------------

def _make_pipeline() -> Pipeline:
    """
    Small traditional text-classification pipeline.

    TF-IDF:
    Converts message text into numeric features.

    LogisticRegression:
    Learns which text patterns are associated with each label.
    """

    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=1,
                    stop_words="english",
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                ),
            ),
        ]
    )


# ---------------------------------------------------------------------
# Local fallback analyzer
# ---------------------------------------------------------------------

class UrgencyAnalyzer:
    """
    Local machine-learning fallback classifier.

    It trains separate models for:
    - urgency
    - category
    - route

    This model is not the primary AI anymore.

    In the hybrid architecture:
        OpenAI = primary classifier
        Local ML = fallback classifier
    """

    def __init__(self):
        self.urgency_pipeline: Optional[
            Pipeline
        ] = None

        self.category_pipeline: Optional[
            Pipeline
        ] = None

        self.route_pipeline: Optional[
            Pipeline
        ] = None

        self.is_trained = False


    # -----------------------------------------------------------------
    # Training
    # -----------------------------------------------------------------

    def train(
        self,
        df: pd.DataFrame,
    ) -> dict:
        """
        Train the local ML models.

        Expected columns:
        - message
        - urgency
        - category
        - route
        """

        info = {
            "n_examples": len(df)
        }

        # -----------------------------
        # Urgency model
        # -----------------------------

        if df["urgency"].nunique() < 2:
            self.urgency_pipeline = None

            info[
                "urgency_trained"
            ] = False

            info[
                "urgency_reason"
            ] = (
                "Only one urgency class "
                "present in training data."
            )

        else:
            self.urgency_pipeline = (
                _make_pipeline()
            )

            self.urgency_pipeline.fit(
                df["message"],
                df["urgency"],
            )

            info[
                "urgency_trained"
            ] = True

            info[
                "urgency_classes"
            ] = sorted(
                df[
                    "urgency"
                ].unique().tolist()
            )


        # -----------------------------
        # Category model
        # -----------------------------

        if (
            df["category"].nunique()
            >= 2
        ):
            self.category_pipeline = (
                _make_pipeline()
            )

            self.category_pipeline.fit(
                df["message"],
                df["category"],
            )

            info[
                "category_trained"
            ] = True

            info[
                "category_classes"
            ] = sorted(
                df[
                    "category"
                ].unique().tolist()
            )

        else:
            self.category_pipeline = None

            info[
                "category_trained"
            ] = False


        # -----------------------------
        # Route model
        # -----------------------------

        if (
            df["route"].nunique()
            >= 2
        ):
            self.route_pipeline = (
                _make_pipeline()
            )

            self.route_pipeline.fit(
                df["message"],
                df["route"],
            )

            info[
                "route_trained"
            ] = True

        else:
            self.route_pipeline = None

            info[
                "route_trained"
            ] = False


        self.is_trained = (
            self.urgency_pipeline
            is not None
        )

        return info


    # -----------------------------------------------------------------
    # Prediction helper
    # -----------------------------------------------------------------

    @staticmethod
    def _predict_with_confidence(
        pipeline: Optional[Pipeline],
        message: str,
        fallback: str,
    ):
        """
        Run a classifier and return:

        predicted label,
        probability of that predicted label.
        """

        if pipeline is None:
            return (
                fallback,
                0.0,
            )

        probabilities = (
            pipeline.predict_proba(
                [message]
            )[0]
        )

        classes = pipeline.classes_

        best_index = (
            probabilities.argmax()
        )

        return (
            classes[best_index],
            float(
                probabilities[
                    best_index
                ]
            ),
        )


    # -----------------------------------------------------------------
    # Analyze one message
    # -----------------------------------------------------------------

    def analyze(
        self,
        message: str,
    ) -> AnalysisResult:

        (
            keyword_score,
            keyword_matches,
            attention_detected,
        ) = _keyword_signal(
            message
        )


        # -------------------------------------------------------------
        # Urgency
        # -------------------------------------------------------------

        if (
            self.urgency_pipeline
            is not None
        ):

            (
                ml_label,
                ml_confidence,
            ) = (
                self._predict_with_confidence(
                    self.urgency_pipeline,
                    message,
                    fallback="Normal",
                )
            )

            # Local ML does most of the work.
            # Keyword risk provides a small additional signal.
            urgency_confidence = min(
                1.0,
                0.8 * ml_confidence
                + 0.2 * keyword_score,
            )

            urgency = ml_label

        else:
            # If the local ML model is unavailable,
            # use the keyword layer only.
            urgency_confidence = (
                keyword_score
            )

            if keyword_score >= 0.66:
                urgency = "Critical"

            elif keyword_score >= 0.33:
                urgency = "High"

            else:
                urgency = "Normal"


        # -------------------------------------------------------------
        # Category
        # -------------------------------------------------------------

        (
            category,
            category_confidence,
        ) = (
            self._predict_with_confidence(
                self.category_pipeline,
                message,
                fallback="Unclassified",
            )
        )


        # -------------------------------------------------------------
        # Route
        # -------------------------------------------------------------

        (
            route,
            _,
        ) = (
            self._predict_with_confidence(
                self.route_pipeline,
                message,
                fallback=category,
            )
        )


        # -------------------------------------------------------------
        # Escalation
        # -------------------------------------------------------------

        escalation = (
            ESCALATION_MAP.get(
                urgency,
                "No",
            )
        )


        # HumanFirst rule:
        #
        # Critical -> immediate human review
        # High     -> priority human review
        # Normal   -> no mandatory review unless confidence is low
        #
        # The hybrid backend will ALSO force human review whenever this
        # local model is being used as a fallback after OpenAI failure.
        escalate_to_human = (
            urgency in {
                "Critical",
                "High",
            }
            or urgency_confidence
            < ESCALATION_CONFIDENCE_THRESHOLD
        )


        # -------------------------------------------------------------
        # Explanation
        # -------------------------------------------------------------

        explanation_parts = []


        if keyword_matches:
            explanation_parts.append(
                "Risk factors identified: "
                + ", ".join(
                    keyword_matches
                )
                + "."
            )


        if attention_detected:
            explanation_parts.append(
                "Uses urgent-sounding "
                "language, but this alone "
                "does not raise the urgency "
                "score."
            )


        if (
            self.urgency_pipeline
            is not None
        ):
            explanation_parts.append(
                f"Local model classified "
                f"as '{urgency}' with "
                f"{urgency_confidence:.0%} "
                f"confidence."
            )

        else:
            explanation_parts.append(
                "Local classifier is not "
                "trained, so a keyword-only "
                "estimate was used."
            )


        if escalate_to_human:
            explanation_parts.append(
                "Flagged for human review."
            )


        # -------------------------------------------------------------
        # Return result
        # -------------------------------------------------------------

        return AnalysisResult(
            message=message,

            urgency=urgency,

            urgency_confidence=(
                urgency_confidence
            ),

            category=category,

            category_confidence=(
                category_confidence
            ),

            route=route,

            escalation=escalation,

            escalate_to_human=(
                escalate_to_human
            ),

            matched_risk_keywords=(
                keyword_matches
            ),

            attention_language_detected=(
                attention_detected
            ),

            explanation=" ".join(
                explanation_parts
            ),
        )


    # -----------------------------------------------------------------
    # Analyze multiple messages
    # -----------------------------------------------------------------

    def analyze_batch(
        self,
        messages: pd.Series,
    ) -> List[AnalysisResult]:

        return [
            self.analyze(message)
            for message
            in messages
        ]


    # -----------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------

    @staticmethod
    def summarize(
        results: List[
            AnalysisResult
        ],
    ) -> dict:

        n = len(results)

        if n == 0:
            return {
                "total_messages": 0,
                "by_urgency": {},
                "by_category": {},
                "escalated_count": 0,
                "escalation_rate": 0.0,
                "average_urgency_confidence":
                    0.0,
            }


        by_urgency = (
            pd.Series(
                [
                    result.urgency
                    for result
                    in results
                ]
            )
            .value_counts()
            .to_dict()
        )


        by_category = (
            pd.Series(
                [
                    result.category
                    for result
                    in results
                ]
            )
            .value_counts()
            .to_dict()
        )


        escalated_count = sum(
            1
            for result
            in results
            if result.escalate_to_human
        )


        average_confidence = (
            sum(
                result.urgency_confidence
                for result
                in results
            )
            / n
        )


        return {
            "total_messages": n,

            "by_urgency":
                by_urgency,

            "by_category":
                by_category,

            "escalated_count":
                escalated_count,

            "escalation_rate":
                round(
                    escalated_count / n,
                    3,
                ),

            "average_urgency_confidence":
                round(
                    average_confidence,
                    3,
                ),
        }