import json
import os
import re
from google import genai
from google.genai import types
from models import LLMAnalysis, UrgencyLevel, Department

SYSTEM_PROMPT = """
You triage incoming public correspondence for a government agency.

Classify the input strictly within the following constraints:

1. Operational Categories & Routing Teams (Must choose exactly one for both category and routing_team):
   - Health & Safety
   - Housing & Utilities
   - Financial Support
   - Licensing & Services
   - General Enquiries

2. Urgency Levels:
   - Critical: Immediate risk to life, acute danger, active homelessness tonight, severe distress.
   - High: Urgent statutory/legal deadlines, impending utility shutoffs, acute financial pressure.
   - Normal: Routine inquiries, status updates, general feedback.

3. Explanation: 1-2 factual sentences explaining why this urgency and department were assigned.

4. Draft Acknowledgement:
   - If Critical: Set draft_acknowledgement to null (automated replies are strictly blocked).
   - If High or Normal: Provide a courteous, neutral acknowledgement that an officer will review the submission.

5. Confidence Score: A float between 0.0 and 1.0 reflecting classification certainty.
"""

def _offline_fallback(content: str) -> LLMAnalysis:
    """Deterministic local fallback when the API is unreachable or key is missing."""
    text = content.lower()
    
    # Category detection
    if any(w in text for w in ["evict", "rent", "landlord", "power", "water", "electricity", "housing"]):
        dept = Department.HOUSING_UTILITIES
    elif any(w in text for w in ["suicide", "harm", "abuse", "hospital", "doctor", "health", "injury"]):
        dept = Department.HEALTH_SAFETY
    elif any(w in text for w in ["grant", "pension", "payment", "bank", "money", "allowance", "food"]):
        dept = Department.FINANCIAL_SUPPORT
    elif any(w in text for w in ["license", "permit", "renew", "registration"]):
        dept = Department.LICENSING_SERVICES
    else:
        dept = Department.GENERAL_ENQUIRIES

    # Urgency detection
    if any(w in text for w in ["kill myself", "suicide", "nowhere to sleep", "tonight", "abuse", "danger"]):
        urgency = UrgencyLevel.CRITICAL
        draft = None
    elif any(w in text for w in ["deadline", "tomorrow", "shut off", "court"]):
        urgency = UrgencyLevel.HIGH
        draft = "Thank you for contacting us. Your time-sensitive inquiry has been logged for priority officer review."
    else:
        urgency = UrgencyLevel.NORMAL
        draft = "Thank you for contacting us. We have received your inquiry and will respond in due course."

    return LLMAnalysis(
        category=dept,
        urgency=urgency,
        reasoning="Classified via deterministic fallback rules.",
        routing_team=dept,
        draft_acknowledgement=draft,
        confidence_score=0.85
    )

def analyze_message(content: str) -> LLMAnalysis:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return _offline_fallback(content)

    try:
        client = genai.Client(api_key=api_key)

        # Allow distressing public reports (self-harm, domestic threats) through to triage
        permissive_safety = [
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                threshold=types.HarmBlockThreshold.BLOCK_NONE,
            ),
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                threshold=types.HarmBlockThreshold.BLOCK_NONE,
            ),
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                threshold=types.HarmBlockThreshold.BLOCK_NONE,
            ),
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                threshold=types.HarmBlockThreshold.BLOCK_NONE,
            ),
        ]

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=content,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                response_mime_type="application/json",
                response_schema=LLMAnalysis,
                safety_settings=permissive_safety,
                temperature=0.0
            )
        )

        # SDK automatically parses into the Pydantic model
        if hasattr(response, "parsed") and response.parsed is not None:
            return response.parsed
        
        # Fallback parsing in case raw text JSON is returned
        data = json.loads(response.text)
        return LLMAnalysis(**data)

    except Exception:
        # Fails safely to offline classification; app.py fail-safe handles unhandled crashes
        return _offline_fallback(content)
