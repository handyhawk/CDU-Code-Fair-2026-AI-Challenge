import json
from openai import OpenAI
from backend.models import LLMAnalysis, UrgencyLevel

client = OpenAI()

SYSTEM_PROMPT = """
You triage incoming public correspondence for a government department.
Tasks:
1. Classify the message into an operational category.
2. Determine urgency: Critical (immediate danger/harm, homelessness, severe distress), High (time-sensitive issues, legal deadlines), or Normal (standard queries, routine feedback).
3. Assign the appropriate destination team.
4. Provide a clear 1-2 sentence explanation of your classification.
5. If Normal or High, draft a neutral acknowledgment. If Critical, set draft_acknowledgement to null (humans must respond).
Respond strictly in JSON matching the requested schema.
"""

def analyze_message(content: str) -> LLMAnalysis:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content}
        ],
        temperature=0.0
    )
    data = json.loads(response.choices[0].message.content)
    return LLMAnalysis(**data)