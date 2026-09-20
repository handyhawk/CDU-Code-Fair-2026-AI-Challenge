from fastapi import FastAPI, HTTPException
from backend.models import TriageRequest, FinalTriageResult, UrgencyLevel, LLMAnalysis
from backend.classifier import analyze_message
from safety_engine import apply_safety_rules

app = FastAPI(title="HumanFirst Triage Engine")

@app.post("/triage", response_model=FinalTriageResult)
def triage_endpoint(request: TriageRequest):
    try:
        # Step 1: AI classification
        ai_result = analyze_message(request.content)
    except Exception as e:
        # Step 2: Fallback Hardening (B3) - Fail open to human escalation[cite: 1]
        ai_result = LLMAnalysis(
            category="System Failure / Unclassified",
            urgency=UrgencyLevel.CRITICAL,
            reasoning=f"AI failure encountered: {str(e)}. Defaulted to human triage.",
            routing_team="Duty Triage Manager",
            draft_acknowledgement=None,
            confidence_score=0.0
        )

    # Step 3: Safety Engine review
    final_result = apply_safety_rules(
        message_id=request.message_id,
        raw_content=request.content,
        ai_result=ai_result
    )
    return final_result