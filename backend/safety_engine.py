"""Independent safety layer for HumanFirst AI.

The classifier (OpenAI or Local ML) makes the first urgency decision.
This module then checks the raw message for high-risk combinations.

Core guarantees:
- It can raise urgency, never lower it.
- Critical/High outcomes always require human review.
- Unknown model labels fail safe to at least High + human review.
- Surface words such as "URGENT" do not raise urgency by themselves.
"""

import re
from dataclasses import dataclass, field
from typing import Any, List


URGENCY_LEVELS = ["Normal", "High", "Critical"]
URGENCY_RANK = {name: i for i, name in enumerate(URGENCY_LEVELS)}

ESCALATION_MAP = {
    "Normal": "No",
    "High": "Yes – Priority",
    "Critical": "Yes – Immediate",
}


@dataclass
class SafetyDecision:
    model_urgency: str
    final_urgency: str
    raised: bool
    escalate_to_human: bool
    reasons: List[str] = field(default_factory=list)
    signals: List[str] = field(default_factory=list)
    negated_terms: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)


# Canonical urgency aliases accepted from either classifier.
_URGENCY_ALIASES = {
    "normal": "Normal",
    "low": "Normal",
    "routine": "Normal",
    "high": "High",
    "priority": "High",
    "elevated": "High",
    "critical": "Critical",
    "urgent": "Critical",
    "emergency": "Critical",
    "immediate": "Critical",
}


# Whole-word / phrase patterns only. This avoids old substring false positives
# such as "blinds" matching "blind" or unrelated words containing fragments.
SIGNAL_PATTERNS = {
    "POWER": [
        r"\bno power\b",
        r"\bpower (?:is |has been |was )?(?:out|off)\b",
        r"\bpower(?:[’'s]+)? (?:has )?been off\b",
        r"\bpower went out\b",
        r"\bpower outage\b",
        r"\bpower cut\b",
        r"\bblackout\b",
        r"\bno electricity\b",
        r"\belectricity (?:is |was |has been )?(?:disconnected|cut off)\b",
    ],
    "MEDICATION": [
        r"\binsulin\b",
        r"\bmedication\b",
        r"\bmedicine\b",
        r"\brefrigerated (?:medicine|medication)\b",
        r"\bprescription\b",
        r"\boxygen concentrator\b",
        r"\boxygen machine\b",
        r"\bhome dialysis\b",
        r"\bdialysis\b",
        r"\bdiabetic\b",
        r"\bdiabetes\b",
        r"\bchemo(?:therapy)?\b",
        r"\bdischarged from hospital\b",
    ],
    "HOUSING": [
        r"\beviction notice\b",
        r"\bevicted\b",
        r"\bevict(?:ion|ed)?\b",
        r"\bhomeless\b",
        r"\bnowhere to (?:go|sleep|stay)\b",
        r"\bsleeping in (?:my|the) car\b",
        r"\blocked (?:me|us) out\b",
        r"\bchanged the locks\b",
        r"\btold (?:me|us) to leave\b",
        r"\btold to leave\b",
    ],
    "THREAT": [
        r"\bthreat(?:en|ened|ening|ens)?\b",
        r"\bviolent\b",
        r"\bviolence\b",
        r"\bhit me\b",
        r"\bhurt me\b",
        r"\bstalk(?:ing|ed|er)?\b",
        r"\bbreak[- ]?in\b",
        r"\bunsafe\b",
        r"\bin danger\b",
    ],
    "VULNERABLE": [
        r"\bbaby\b",
        r"\binfant\b",
        r"\bnewborn\b",
        r"\bchild\b",
        r"\bchildren\b",
        r"\bkids?\b",
        r"\belderly\b",
        r"\bpregnant\b",
        r"\bwheelchair user\b",
        r"\bwheelchair\b",
        r"\bdisabled\b",
        r"\bdisability\b",
        r"\blives? alone\b",
    ],
    "EXTREME_HEAT": [
        r"\bheatwave\b",
        r"\bextreme heat\b",
        r"\bvery hot\b",
    ],
    "SELF_HARM": [
        r"\bdon['’]?t want to live\b",
        r"\bdo not want to live\b",
        r"\bkill myself\b",
        r"\bhurt myself\b",
        r"\bend my life\b",
        r"\bsuicid(?:e|al)\b",
        r"\bself[- ]?harm\b",
    ],
}


# Negation phrases that should cancel a nearby safety term.
# Deliberately NOT included: "don't have" / "do not have" because
# "I don't have my insulin" is a risk, not a negation of the risk.
_NEGATION_PREFIXES = [
    r"\bnot on(?: any)?\s+",
    r"\bnot taking\s+",
    r"\bdon['’]?t take\s+",
    r"\bdo not take\s+",
    r"\bno longer\s+",
    r"\bno one (?:is |was )?",
    r"\bnobody (?:is |was )?",
    r"\bnot (?:being |currently )?",
]


def _normalise_urgency(value: Any) -> tuple[str, bool]:
    if isinstance(value, str):
        key = value.strip().lower()
        if key in _URGENCY_ALIASES:
            return _URGENCY_ALIASES[key], True
    return "High", False  # fail safe for missing/unknown labels


def _is_negated(text: str, start: int) -> bool:
    """Return True when the matched risk term is directly negated nearby."""
    left = text[max(0, start - 45):start]
    return any(re.search(pattern + r"[^.!?]{0,20}$", left) for pattern in _NEGATION_PREFIXES)


def _detect_signals(message: str) -> tuple[set[str], List[str]]:
    text = message.lower()
    signals: set[str] = set()
    negated_terms: List[str] = []

    for signal, patterns in SIGNAL_PATTERNS.items():
        for pattern in patterns:
            for match in re.finditer(pattern, text, flags=re.IGNORECASE):
                term = match.group(0).strip()
                if signal != "SELF_HARM" and _is_negated(text, match.start()):
                    if term not in negated_terms:
                        negated_terms.append(term)
                    continue
                signals.add(signal)
                break
            if signal in signals:
                break

    return signals, negated_terms


def _raise_target(current: str, requested: str) -> str:
    """Return the more urgent label. Safety is never allowed to lower urgency."""
    return requested if URGENCY_RANK[requested] > URGENCY_RANK[current] else current


def evaluate(message: str, model_urgency: Any) -> SafetyDecision:
    """Evaluate a message independently of the classifier output."""
    base, label_valid = _normalise_urgency(model_urgency)
    signals, negated_terms = _detect_signals(message or "")

    final = base
    reasons: List[str] = []
    notes: List[str] = []

    if not label_valid:
        reasons.append("UNKNOWN_MODEL_LABEL")
        notes.append("Classifier urgency was missing or unrecognised; failed safe to High.")

    # Direct critical indicator.
    if "SELF_HARM" in signals:
        final = _raise_target(final, "Critical")
        reasons.append("SELF_HARM")
        notes.append("Direct self-harm or suicide-related language detected.")

    # High-risk combinations. These are intentionally contextual: one word alone
    # should not automatically become Critical.
    if {"POWER", "MEDICATION"}.issubset(signals):
        final = _raise_target(final, "Critical")
        reasons.append("POWER+MEDICATION")
        notes.append("Essential power loss may affect medication or medical equipment.")

    if {"HOUSING", "MEDICATION"}.issubset(signals):
        final = _raise_target(final, "Critical")
        reasons.append("HOUSING+MEDICAL")
        notes.append("Housing crisis is combined with a medical vulnerability.")

    if {"THREAT", "VULNERABLE"}.issubset(signals):
        final = _raise_target(final, "Critical")
        reasons.append("THREAT+VULNERABLE")
        notes.append("Threat or violence involves a vulnerable person.")

    if {"POWER", "EXTREME_HEAT", "VULNERABLE"}.issubset(signals):
        final = _raise_target(final, "Critical")
        reasons.append("POWER+HEAT+VULNERABLE")
        notes.append("Power loss during extreme heat affects a vulnerable person.")

    # Serious single circumstances can justify priority review, but do not jump
    # straight to Critical without a dangerous combination.
    if final == "Normal":
        if "THREAT" in signals:
            final = "High"
            reasons.append("ACTIVE_THREAT")
            notes.append("Threat or violence language requires priority human review.")
        elif "HOUSING" in signals:
            final = "High"
            reasons.append("HOUSING_CRISIS")
            notes.append("Housing instability requires priority human review.")
        elif "POWER" in signals:
            final = "High"
            reasons.append("ESSENTIAL_SERVICE_LOSS")
            notes.append("Loss of an essential service requires priority human review.")

    raised = URGENCY_RANK[final] > URGENCY_RANK[base]
    escalate_to_human = final in {"High", "Critical"} or not label_valid

    return SafetyDecision(
        model_urgency=base,
        final_urgency=final,
        raised=raised,
        escalate_to_human=escalate_to_human,
        reasons=reasons,
        signals=sorted(signals),
        negated_terms=negated_terms,
        notes=notes,
    )


def _get(result: Any, key: str, default=None):
    return result.get(key, default) if isinstance(result, dict) else getattr(result, key, default)


def _set(result: Any, key: str, value: Any) -> None:
    if isinstance(result, dict):
        result[key] = value
    else:
        setattr(result, key, value)


def apply_safety(message: str, result: Any):
    """Apply the independent safety decision to a classifier result.

    Supports both:
    - dictionaries returned by the OpenAI path
    - AnalysisResult-like objects returned by the Local ML path
    """
    original_label = _get(result, "urgency")
    decision = evaluate(message, original_label)

    # Work on a shallow copy for dictionaries so the classifier output is not
    # unexpectedly mutated elsewhere. Objects are updated in place.
    if isinstance(result, dict):
        result = dict(result)

    existing_review = bool(_get(result, "escalate_to_human", False))
    final_review = existing_review or decision.escalate_to_human

    _set(result, "urgency", decision.final_urgency)
    _set(result, "escalation", ESCALATION_MAP[decision.final_urgency])
    _set(result, "escalate_to_human", final_review)

    safety_payload = {
        "checked": True,
        "model_urgency": decision.model_urgency,
        "final_urgency": decision.final_urgency,
        "raised": decision.raised,
        "reasons": decision.reasons,
        "signals": decision.signals,
        "negated_terms": decision.negated_terms,
    }
    _set(result, "safety", safety_payload)
    _set(result, "safety_notes", decision.notes)

    # Preserve any pre-existing matched risk keywords and add the safety signals.
    existing_risks = list(_get(result, "matched_risk_keywords", []) or [])
    combined_risks = existing_risks[:]
    for signal in decision.signals:
        if signal not in combined_risks:
            combined_risks.append(signal)
    _set(result, "matched_risk_keywords", combined_risks)

    explanation = str(_get(result, "explanation", "") or "").strip()
    if decision.raised:
        safety_text = (
            f"Safety Engine raised urgency from {decision.model_urgency} "
            f"to {decision.final_urgency}"
        )
        if decision.reasons:
            safety_text += " due to " + ", ".join(decision.reasons)
        safety_text += "."
        explanation = f"{explanation} {safety_text}".strip()
    elif "UNKNOWN_MODEL_LABEL" in decision.reasons:
        explanation = (
            f"{explanation} Safety Engine could not validate the classifier urgency and "
            "failed safe to High with human review."
        ).strip()

    _set(result, "explanation", explanation)
    return result
