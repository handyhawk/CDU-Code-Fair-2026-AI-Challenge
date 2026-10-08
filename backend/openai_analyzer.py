from enum import Enum

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

from data_loader import ESCALATION_MAP


load_dotenv()

class Urgency(str, Enum):
    critical = "Critical"
    high = "High"
    normal = "Normal"


class Category(str, Enum):
    health_safety = "Health & Safety"
    housing_utilities = "Housing & Utilities"
    financial_support = "Financial Support"
    licensing_services = "Licensing & Services"
    general_enquiries = "General Enquiries"


class OpenAIClassification(BaseModel):
    urgency: Urgency
    category: Category
    route: Category
    explanation: str


SYSTEM_PROMPT = """
You are the primary AI classifier for HumanFirst AI, a government inbox
triage prototype.

Classify each incoming message.

URGENCY

Critical:
Immediate or potentially immediate threat involving health, safety,
shelter, essential services or severe vulnerability.

High:
Important or time-sensitive issue requiring priority human review,
but without an immediate critical threat.

Normal:
Routine enquiry, administrative request or non-urgent service issue.

CATEGORY

Choose exactly one:
- Health & Safety
- Housing & Utilities
- Financial Support
- Licensing & Services
- General Enquiries

ROUTE

Choose exactly one team from the same five options.

Category describes what the message is mainly about.
Route describes which team should handle it.

Category and route do not have to be the same.

EXPLANATION

Give one short, neutral sentence explaining the classification.

IMPORTANT RULES

Do not increase urgency merely because the message contains words such
as "urgent", "ASAP", "emergency", capital letters or many exclamation marks.

Base urgency on the actual circumstances described.

Do not make final decisions for the government agency.
Human staff remain responsible for the final decision.
"""


def classify_message(message: str) -> OpenAIClassification:
    """Classify a message using the primary OpenAI model."""
    client = OpenAI()

    response = client.responses.parse(
        model="gpt-5-mini",
        input=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": message,
            },
        ],
        text_format=OpenAIClassification,
    )

    if response.output_parsed is None:
        raise RuntimeError("OpenAI returned no structured classification.")

    return response.output_parsed


def to_backend_dict(
    message: str,
    result: OpenAIClassification,
) -> dict:
    urgency = result.urgency.value

    return {
        "message": message,
        "urgency": urgency,


        "urgency_confidence": None,

        "category": result.category.value,
        "category_confidence": None,
        "route": result.route.value,

        "escalation": ESCALATION_MAP.get(urgency, "No"),


        "escalate_to_human": urgency in {"Critical", "High"},

        "matched_risk_keywords": [],
        "attention_language_detected": False,

        "explanation": result.explanation,

        "analysis_mode": "OpenAI",
        "provisional": False,
    }


if __name__ == "__main__":
    test_message = (
        "My electricity has been disconnected and my insulin "
        "needs refrigeration."
    )

    result = classify_message(test_message)
    backend_result = to_backend_dict(test_message, result)

    print(backend_result)
