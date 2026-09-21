import re
from typing import List, Tuple
from models import UrgencyLevel, LLMAnalysis, FinalTriageResult
 
CRITICAL_TRIGGERS = [
    r"\b(suicide|self-harm|kill myself|end my life)\b",
    r"\b(evict(ion|ed)? today|nowhere to sleep tonight|sleeping in car)\b",
    r"\b(abuse|assault|violence|immediate danger|threat)\b"
]
 
HIGH_TRIGGERS = [
    r"\b(deadline|court date|hearing tomorrow|urgent medicine)\b"
]
 
def apply_safety_rules(
    message_id: str, 
    raw_content: str, 
    ai_result: LLMAnalysis
) -> FinalTriageResult:
    triggers: List[str] = []
    
    # 1. Regex scanning for critical vulnerability flags
    for pattern in CRITICAL_TRIGGERS:
        if re.search(pattern, raw_content, re.IGNORECASE):
            triggers.append(f"Critical trigger detected: '{pattern}'")
 
    for pattern in HIGH_TRIGGERS:
        if re.search(pattern, raw_content, re.IGNORECASE):
            triggers.append(f"High-priority trigger detected: '{pattern}'")
 
    # 2. Urgency Monotonicity Enforcement (Safety Engine only elevates)[cite: 1]
    final_urgency = ai_result.urgency
    if triggers:
        if any("Critical" in t for t in triggers):
            final_urgency = UrgencyLevel.CRITICAL
        elif final_urgency == UrgencyLevel.NORMAL:
            final_urgency = UrgencyLevel.HIGH
 
    # 3. Uncertainty Flagging
    if ai_result.confidence_score < 0.65:
        triggers.append("Low AI confidence score")
 
    # 4. Mandatory Human Escalation Rule
    # Critical messages ALWAYS escalate immediately and suppress automated responses
    escalate = final_urgency == UrgencyLevel.CRITICAL or len(triggers) > 0
    final_draft = None if escalate else ai_result.draft_acknowledgement
 
    return FinalTriageResult(
        message_id=message_id,
        final_urgency=final_urgency,
        final_route="Emergency Crisis Team" if final_urgency == UrgencyLevel.CRITICAL else ai_result.routing_team,
        escalate_to_human=escalate,
        safety_triggers=triggers,
        explanation=ai_result.reasoning,
        draft_acknowledgement=final_draft
    )
 