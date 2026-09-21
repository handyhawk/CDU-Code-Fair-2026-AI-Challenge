from pydantic import BaseModel, Field
from typing import Optional, List
from enum import Enum
 
class UrgencyLevel(str, Enum):
    NORMAL = "Normal"
    HIGH = "High"
    CRITICAL = "Critical"
 
class TriageRequest(BaseModel):
    message_id: str
    content: str
    sender_metadata: Optional[dict] = None
 
class LLMAnalysis(BaseModel):
    category: str
    urgency: UrgencyLevel
    reasoning: str
    routing_team: str
    draft_acknowledgement: Optional[str] = None
    confidence_score: float = Field(..., ge=0.0, le=1.0)
 
class FinalTriageResult(BaseModel):
    message_id: str
    final_urgency: UrgencyLevel
    final_route: str
    escalate_to_human: bool
    safety_triggers: List[str]
    explanation: str
    draft_acknowledgement: Optional[str]
 