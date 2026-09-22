from pydantic import BaseModel, Field
from typing import Optional, List
from enum import Enum

class UrgencyLevel(str, Enum):
    NORMAL = "Normal"
    HIGH = "High"
    CRITICAL = "Critical"

class Department(str, Enum):
    HEALTH_SAFETY = "Health & Safety"
    HOUSING_UTILITIES = "Housing & Utilities"
    FINANCIAL_SUPPORT = "Financial Support"
    LICENSING_SERVICES = "Licensing & Services"
    GENERAL_ENQUIRIES = "General Enquiries"

class TriageRequest(BaseModel):
    message_id: str
    content: str
    sender_metadata: Optional[dict] = None

class LLMAnalysis(BaseModel):
    category: Department
    urgency: UrgencyLevel
    reasoning: str
    routing_team: Department
    draft_acknowledgement: Optional[str] = None
    confidence_score: float = Field(..., ge=0.0, le=1.0)

class FinalTriageResult(BaseModel):
    message_id: str
    category: Department
    final_urgency: UrgencyLevel
    final_route: Department
    explanation: str
    escalate_to_human: bool
    safety_triggers: List[str]
    confidence_score: float
    draft_acknowledgement: Optional[str] = None