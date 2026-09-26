"""
data_loader.py
"""

import pandas as pd

COLUMN_ALIASES = {
    "message": "message",
    "text": "message",
    "message_text": "message",
    "content": "message",

    "urgency": "urgency",
    "expected_urgency": "urgency",
    "label": "urgency",
    "urgency_label": "urgency",

    "category": "category",
    "expected_category": "category",

    "route": "route",
    "expected_route": "route",

    "escalation": "escalation",
    "expected_escalation": "escalation",

    "reason": "reason",
    "risk_trigger": "reason",
    "justification": "reason",
    "notes": "reason",
}

REQUIRED_COLUMNS = ["message", "urgency"]

# Canonical urgency tiers, ordered from lowest to highest.
URGENCY_LEVELS = ["Normal", "High", "Critical"]

# A predicted urgency maps deterministically onto a business escalation
# tier — this mirrors the 1:1 relationship observed in the labeled data
# (Critical -> Immediate, High -> Priority, Normal -> No escalation).
ESCALATION_MAP = {
    "Critical": "Yes – Immediate",
    "High": "Yes – Priority",
    "Normal": "No",
}


def _normalise_headers(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {}
    for col in df.columns:
        key = col.strip().lower().replace(" ", "_")
        if key in COLUMN_ALIASES:
            rename_map[col] = COLUMN_ALIASES[key]
    return df.rename(columns=rename_map)


def _normalise_urgency_value(value) -> str:
    """
    Collapse whatever's in the urgency column onto one of the three
    canonical tiers. Accepts case variants and a few common synonyms so
    the loader tolerates minor formatting differences between exports.
    """
    if pd.isna(value):
        return "Normal"

    text = str(value).strip().lower()
    if text in {"critical", "urgent", "emergency", "immediate"}:
        return "Critical"
    if text in {"high", "priority", "elevated"}:
        return "High"
    return "Normal"


def load_training_data(csv_path: str) -> pd.DataFrame:
    """
    Load and validate the labeled training CSV. Raises a clear
    ValueError if required columns are missing, rather than failing
    deep inside model training with a cryptic error.
    """
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    df = _normalise_headers(df)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"CSV is missing required column(s): {missing}. "
            f"Found columns: {list(df.columns)}."
        )

    df = df.dropna(subset=["message"]).copy()
    df["message"] = df["message"].astype(str).str.strip()
    df = df[df["message"] != ""]

    df["urgency"] = df["urgency"].apply(_normalise_urgency_value)

    # category / route / reason are optional — fill with a placeholder
    # if the CSV doesn't have them yet, rather than erroring out. This
    # lets the backend still train an urgency-only model on a simpler
    # CSV if that's all that's available at some point.
    for optional_col, default in [
        ("category", "Unclassified"),
        ("route", "Unclassified"),
        ("escalation", ""),
        ("reason", ""),
    ]:
        if optional_col not in df.columns:
            df[optional_col] = default
        else:
            df[optional_col] = df[optional_col].fillna(default).astype(str).str.strip()

    return df[["message", "urgency", "category", "route", "escalation", "reason"]].reset_index(drop=True)


def load_inbox_csv(csv_path: str) -> pd.DataFrame:
    """
    Load a CSV of *incoming* messages to triage (no labels expected).
    Used by the /analyze batch endpoint.
    """
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    df = _normalise_headers(df)

    if "message" not in df.columns:
        raise ValueError(
            f"CSV needs a 'message' (or 'text') column. Found: {list(df.columns)}"
        )

    df = df.dropna(subset=["message"]).copy()
    df["message"] = df["message"].astype(str).str.strip()
    df = df[df["message"] != ""]
    return df.reset_index(drop=True)