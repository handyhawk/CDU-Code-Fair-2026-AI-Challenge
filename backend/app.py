from fastapi import FastAPI
from models import TriageRequest, FinalTriageResult, UrgencyLevel, Department, LLMAnalysis
from classifier import analyze_message
from safety_engine import apply_safety_rules

app = FastAPI(
    title="HumanFirst Triage Engine",
    version="2.0.0"
)

@app.post("/triage", response_model=FinalTriageResult)
def triage_endpoint(request: TriageRequest):
    try:
        ai_result = analyze_message(request.content)
    except Exception as e:
        # Fallback Hardening (B3): Fail-open to human oversight
        ai_result = LLMAnalysis(
            category=Department.GENERAL_ENQUIRIES,
            urgency=UrgencyLevel.CRITICAL,
            reasoning=f"System error encountered during automated triage: {str(e)}. Defaulted to human review.",
            routing_team=Department.GENERAL_ENQUIRIES,
            draft_acknowledgement=None,
            confidence_score=0.0
        )

    final_result = apply_safety_rules(
        message_id=request.message_id,
        raw_content=request.content,
        ai_result=ai_result
    )
    
    return final_result