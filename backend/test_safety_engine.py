"""Tests for safety_engine.py.  Run with:  pytest -v test_safety_engine.py"""
from dataclasses import dataclass, field
from typing import List

import pytest

from safety_engine import URGENCY_LEVELS, URGENCY_RANK, apply_safety, evaluate

# A stand-in with the same fields as the project's AnalysisResult.
@dataclass
class FakeResult:
    message: str
    urgency: str
    escalation: str = "No"
    escalate_to_human: bool = False
    explanation: str = ""
    matched_risk_keywords: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------
# 1. Required risk combinations -> Critical, even if the model said Normal
# ---------------------------------------------------------------------

POWER_MEDICATION = [
    "No power since Friday and my insulin is getting warm",
    "Power outage in our street, my dad's oxygen concentrator won't run",
    "The power's been off two days. I keep my medication in the fridge",
    "Blackout last night - mum is on home dialysis",
    "electricity disconnected and I ran out of medication",
]
HOUSING_MEDICAL = [
    "We got an eviction notice and I'm on dialysis",
    "I'm homeless right now and I'm diabetic",
    "Landlord changed the locks, my insulin is still inside",
    "Sleeping in my car after chemo, I have nowhere to go",
    "Told to leave by Friday, I was just discharged from hospital",
]
THREAT_VULNERABLE = [
    "My ex threatened me again and my kids are here",
    "My neighbour is threatening my elderly mother",
    "I'm pregnant and my partner hit me last night",
    "Someone keeps stalking me, I'm a wheelchair user and live alone",
    "There was a break-in and my baby is in the house",
]


@pytest.mark.parametrize("msg", POWER_MEDICATION)
def test_power_plus_medication_is_critical(msg):
    d = evaluate(msg, "Normal")
    assert d.final_urgency == "Critical"
    assert "POWER+MEDICATION" in d.reasons
    assert d.escalate_to_human


@pytest.mark.parametrize("msg", HOUSING_MEDICAL)
def test_housing_plus_medical_is_critical(msg):
    d = evaluate(msg, "Normal")
    assert d.final_urgency == "Critical"
    assert "HOUSING+MEDICAL" in d.reasons


@pytest.mark.parametrize("msg", THREAT_VULNERABLE)
def test_threat_plus_vulnerable_person_is_critical(msg):
    d = evaluate(msg, "Normal")
    assert d.final_urgency == "Critical"
    assert "THREAT+VULNERABLE" in d.reasons


def test_power_heat_and_elderly_is_critical():
    d = evaluate("Power went out in the heatwave and my elderly father lives alone", "Normal")
    assert d.final_urgency == "Critical"


def test_self_harm_alone_is_critical():
    assert evaluate("I don't want to live anymore", "Normal").final_urgency == "Critical"


# ---------------------------------------------------------------------
# 2. A single factor alone should NOT jump to Critical
# ---------------------------------------------------------------------

@pytest.mark.parametrize("msg", [
    "There's a power outage on my street, when will it be fixed?",
    "Can I get a repeat prescription for my medication?",
    "I received an eviction notice, what are my rights?",
])
def test_single_factor_does_not_force_critical(msg):
    assert evaluate(msg, "Normal").final_urgency != "Critical"


# ---------------------------------------------------------------------
# 3. Negation and old false positives must not raise
# ---------------------------------------------------------------------

@pytest.mark.parametrize("msg", [
    "I'm not on any medication, just a question about my power bill",
    "No one is threatening me, I just want to change my address",
    "He is no longer violent and we're doing fine",
    # false positives from the old substring matcher:
    "Avocado prices are up, I've begun budgeting",
    "My pipeline project deadline is Friday",
    "I had a cramp in my leg",
    "The blinds in my room are broken",
    "URGENT!!! please reset my password",
])
def test_benign_or_negated_messages_are_not_raised(msg):
    d = evaluate(msg, "Normal")
    assert d.final_urgency == "Normal", (d.reasons, d.signals)
    assert not d.raised


def test_negated_term_is_reported():
    d = evaluate("I'm not on any medication", "Normal")
    assert "medication" in d.negated_terms


def test_lack_of_medication_is_not_treated_as_negation():
    d = evaluate("Power cut and I don't have my insulin", "Normal")
    assert d.final_urgency == "Critical"


# ---------------------------------------------------------------------
# 4. The core guarantee: never lowers urgency, never turns review off
# ---------------------------------------------------------------------

CORPUS = (POWER_MEDICATION + HOUSING_MEDICAL + THREAT_VULNERABLE + [
    "Hello, how do I update my email address?",
    "URGENT!!! please reset my password",
    "I'm not on any medication",
    "",
])


@pytest.mark.parametrize("label", URGENCY_LEVELS)
@pytest.mark.parametrize("msg", CORPUS)
def test_never_lowers_urgency(msg, label):
    d = evaluate(msg, label)
    assert URGENCY_RANK[d.final_urgency] >= URGENCY_RANK[label]


@pytest.mark.parametrize("msg", CORPUS)
def test_never_turns_human_review_off(msg):
    r = FakeResult(message=msg, urgency="Normal", escalate_to_human=True)
    assert apply_safety(msg, r).escalate_to_human is True


def test_critical_from_model_stays_critical_on_benign_text():
    r = FakeResult(message="hi", urgency="Critical", escalation="Yes – Immediate", escalate_to_human=True)
    out = apply_safety("hi", r)
    assert out.urgency == "Critical" and out.escalate_to_human


# ---------------------------------------------------------------------
# 5. Both classifier paths: local ML object and OpenAI-style dict
# ---------------------------------------------------------------------

MSG = "No power since Friday and my insulin is getting warm"


def test_local_ml_result_object_is_raised():
    out = apply_safety(MSG, FakeResult(message=MSG, urgency="Normal"))
    assert out.urgency == "Critical"
    assert out.escalate_to_human
    assert out.escalation != "No"
    assert "Safety Engine raised" in out.explanation
    assert out.safety["model_urgency"] == "Normal"


def test_openai_dict_result_is_raised():
    out = apply_safety(MSG, {"urgency": "normal", "category": "Energy"})
    assert out["urgency"] == "Critical"
    assert out["escalate_to_human"] is True


@pytest.mark.parametrize("bad_label", [None, "", "Medium-ish", 3, "unknown"])
def test_unknown_model_label_fails_safe(bad_label):
    out = apply_safety("Hello", {"urgency": bad_label})
    assert out["urgency"] in ("High", "Critical")
    assert out["escalate_to_human"] is True
    assert "UNKNOWN_MODEL_LABEL" in out["safety"]["reasons"]


def test_label_aliases_are_understood():
    assert evaluate("Hello", "critical").final_urgency == "Critical"
    assert evaluate("Hello", "LOW").final_urgency == "Normal"