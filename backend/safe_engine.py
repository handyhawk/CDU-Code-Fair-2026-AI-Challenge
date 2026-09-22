import re
from typing import List
from models import UrgencyLevel, Department, LLMAnalysis, FinalTriageResult

# Single critical regex patterns
CRITICAL_STANDALONE = [
    r"\b(suicide|self-harm|kill myself|end my life)\b",
    r"\b(evict(ion|ed)? today|nowhere to sleep tonight|sleeping in car)\b",
    r"\b(domestic violence|physical abuse|assault|in immediate danger|threatened my life)\b"
]

HIGH_STANDALONE = [
    r"\b(statutory deadline|appeal deadline|court date|hearing tomorrow|tribunal)\b",
    r"\b(urgent prescription|running out of medication|insulin|oxygen tank)\b",
    r"\b(disconnection notice|power cut imminent|shut off notice)\b"
]

# Compound Risk Logic
def check_compound_risks(text: str) -> List[str]:
    flags = []
    
    # 1. Electricity loss + Medical dependency -> Critical
    has_power_loss = bool(re.search(r"\b(no electricity|power cut|power (is )?off|shut off|disconnection)\b", text, re.I))
    has_medical_need = bool(re.search(r"\b(ventilator|dialysis|oxygen machine|life support|refrigerated medication|cpap)\b", text, re.I))
    if has_power_loss and has_medical_need:
        flags.append("Combined Risk (Critical): Power loss coupled with critical medical dependency")

    # 2. Homelessness / Eviction + Vulnerable Person -> Critical
    has_housing_insecurity = bool(re.search(r"\b(homeless|evict(ion|ed)?|no shelter|street|couch surfing)\b", text, re.I))
    has_vulnerable_person = bool(re.search(r"\b(baby|infant|toddler|children|elderly|frail|disabled|pregnant)\b", text, re.I))
    if has_housing_insecurity and has_vulnerable_person:
        flags.append("Combined Risk (Critical): Housing loss involving vulnerable dependent(s)")

    # 3. Financial Hardship + Acute Deprivation (No food/essential needs) -> Critical
    has_poverty = bool(re.search(r"\b(no money|broke|zero balance|bank account empty|penniless)\b", text, re.I))
    has_starvation = bool(re.search(r"\b(no food|starving|cannot feed|baby formula|essential needs)\b", text, re.I))
    if has_poverty and has_starvation:
        flags.append("Combined Risk (Critical): Destitution with immediate food or essential deprivation")

    return flags

def apply_safety_rules(
    message_id: str, 
    raw_content: str, 
    ai_result: LLMAnalysis
) -> FinalTriageResult:
    triggers: List[str] = []
    
    # 1. Scan Compound Patterns
    compound_flags = check_compound_risks(raw_content)
    triggers.extend(compound_flags)

    # 2. Scan Standalone Patterns
    for pattern in CRITICAL_STANDALONE:
        if re.search(pattern, raw_content, re.IGNORECASE):
            triggers.append(f"Critical trigger: '{pattern}'")

    for pattern in HIGH_STANDALONE:
        if re.search(pattern, raw_content, re.IGNORECASE):
            triggers.append(f"High-priority trigger: '{pattern}'")

    # 3. Urgency Monotonicity (Can elevate, never lower)
    final_urgency = ai_result.urgency

    has_critical_trigger = any("Critical" in t for t in triggers)
    has_high_trigger = any("High-priority" in t for t in triggers)

    if has_critical_trigger:
        final_urgency = UrgencyLevel.CRITICAL
    elif has_high_trigger and final_urgency == UrgencyLevel.NORMAL:
        final_urgency = UrgencyLevel.HIGH

    # 4. Confidence Thresholding
    if ai_result.confidence_score < 0.65:
        triggers.append(f"Low AI confidence score ({ai_result.confidence_score:.2f})")

    # 5. Escalation Rule:
    # Critical -> Immediate human review
    # High -> Priority human review
    # Low-confidence / Safety trigger -> Human review
    # Normal (no triggers, high confidence) -> Standard queue
    escalate_to_human = (
        final_urgency in [UrgencyLevel.CRITICAL, UrgencyLevel.HIGH]
        or len(triggers) > 0
        or ai_result.confidence_score < 0.65
    )

    # 6. Acknowledgement Gate (The Trust Twist):
    # Critical -> Suppressed (None)
    # High / Normal -> Draft retained for human review / dispatch
    if final_urgency == UrgencyLevel.CRITICAL:
        final_draft = None
    else:
        final_draft = ai_result.draft_acknowledgement

    return FinalTriageResult(
        message_id=message_id,
        category=ai_result.category,
        final_urgency=final_urgency,
        final_route=ai_result.routing_team,
        explanation=ai_result.reasoning,
        escalate_to_human=escalate_to_human,
        safety_triggers=triggers,
        confidence_score=ai_result.confidence_score,
        draft_acknowledgement=final_draft
    )
