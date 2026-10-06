import json
import os
import sqlite3
from datetime import datetime, timezone
from typing import Optional


DEFAULT_DB_PATH = os.path.join(
    os.path.dirname(__file__),
    "data",
    "cases.db",
)


SCHEMA = """
CREATE TABLE IF NOT EXISTS cases (
    id                          INTEGER PRIMARY KEY AUTOINCREMENT,
    message                     TEXT NOT NULL,
    urgency                     TEXT NOT NULL,
    urgency_confidence          REAL,
    category                    TEXT,
    category_confidence         REAL,
    route                       TEXT,
    escalation                  TEXT,
    escalate_to_human           INTEGER NOT NULL DEFAULT 0,
    matched_risk_keywords       TEXT,
    attention_language_detected INTEGER NOT NULL DEFAULT 0,
    explanation                 TEXT,
    analysis_mode               TEXT NOT NULL DEFAULT 'Local ML',
    provisional                 INTEGER NOT NULL DEFAULT 0,
    source                      TEXT NOT NULL DEFAULT 'single',
    created_at                  TEXT NOT NULL,
    human_reviewed              INTEGER NOT NULL DEFAULT 0,
    human_urgency               TEXT,
    human_notes                 TEXT,
    reviewed_at                 TEXT
);
"""


def get_connection(
    db_path: str = DEFAULT_DB_PATH,
) -> sqlite3.Connection:
    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    return conn


def init_db(
    db_path: str = DEFAULT_DB_PATH,
) -> None:
    """
    Create the database if it does not exist.

    Also adds newer HumanFirst fields to an older database without
    requiring the existing database to be deleted.
    """

    conn = get_connection(db_path)

    try:
        conn.execute(SCHEMA)

        existing_columns = {
            row["name"]
            for row in conn.execute(
                "PRAGMA table_info(cases)"
            ).fetchall()
        }

        if "analysis_mode" not in existing_columns:
            conn.execute(
                """
                ALTER TABLE cases
                ADD COLUMN analysis_mode
                TEXT NOT NULL DEFAULT 'Local ML'
                """
            )

        if "provisional" not in existing_columns:
            conn.execute(
                """
                ALTER TABLE cases
                ADD COLUMN provisional
                INTEGER NOT NULL DEFAULT 0
                """
            )

        conn.commit()

    finally:
        conn.close()


def _row_to_dict(row: sqlite3.Row) -> dict:
    d = dict(row)

    d["escalate_to_human"] = bool(
        d["escalate_to_human"]
    )

    d["attention_language_detected"] = bool(
        d["attention_language_detected"]
    )

    d["human_reviewed"] = bool(
        d["human_reviewed"]
    )

    d["provisional"] = bool(
        d.get("provisional", 0)
    )

    d["matched_risk_keywords"] = json.loads(
        d["matched_risk_keywords"] or "[]"
    )

    return d


def insert_case(
    analysis: dict,
    source: str = "single",
    db_path: str = DEFAULT_DB_PATH,
) -> dict:
    """
    Store one analyzed case.

    The analysis may come from:
    - OpenAI primary classifier
    - Local ML fallback
    """

    conn = get_connection(db_path)

    try:
        cursor = conn.execute(
            """
            INSERT INTO cases (
                message,
                urgency,
                urgency_confidence,
                category,
                category_confidence,
                route,
                escalation,
                escalate_to_human,
                matched_risk_keywords,
                attention_language_detected,
                explanation,
                analysis_mode,
                provisional,
                source,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                analysis["message"],
                analysis["urgency"],
                analysis.get("urgency_confidence"),
                analysis.get("category"),
                analysis.get("category_confidence"),
                analysis.get("route"),
                analysis.get("escalation"),
                int(
                    bool(
                        analysis.get(
                            "escalate_to_human"
                        )
                    )
                ),
                json.dumps(
                    analysis.get(
                        "matched_risk_keywords",
                        [],
                    )
                ),
                int(
                    bool(
                        analysis.get(
                            "attention_language_detected"
                        )
                    )
                ),
                analysis.get("explanation"),
                analysis.get(
                    "analysis_mode",
                    "Local ML",
                ),
                int(
                    bool(
                        analysis.get(
                            "provisional",
                            False,
                        )
                    )
                ),
                source,
                datetime.now(
                    timezone.utc
                ).isoformat(),
            ),
        )

        conn.commit()

        new_id = cursor.lastrowid

        row = conn.execute(
            "SELECT * FROM cases WHERE id = ?",
            (new_id,),
        ).fetchone()

        return _row_to_dict(row)

    finally:
        conn.close()


def insert_cases_batch(
    analyses: list[dict],
    source: str = "batch",
    db_path: str = DEFAULT_DB_PATH,
) -> list[dict]:

    return [
        insert_case(
            analysis,
            source=source,
            db_path=db_path,
        )
        for analysis in analyses
    ]


def get_cases(
    urgency: Optional[str] = None,
    category: Optional[str] = None,
    escalate_to_human: Optional[bool] = None,
    limit: Optional[int] = None,
    db_path: str = DEFAULT_DB_PATH,
) -> list[dict]:

    conn = get_connection(db_path)

    try:
        clauses = []
        params = []

        if urgency:
            clauses.append("urgency = ?")
            params.append(urgency)

        if category:
            clauses.append("category = ?")
            params.append(category)

        if escalate_to_human is not None:
            clauses.append(
                "escalate_to_human = ?"
            )
            params.append(
                int(escalate_to_human)
            )

        query = "SELECT * FROM cases"

        if clauses:
            query += (
                " WHERE "
                + " AND ".join(clauses)
            )

        query += " ORDER BY id DESC"

        if limit:
            query += " LIMIT ?"
            params.append(limit)

        rows = conn.execute(
            query,
            params,
        ).fetchall()

        return [
            _row_to_dict(row)
            for row in rows
        ]

    finally:
        conn.close()


def get_case(
    case_id: int,
    db_path: str = DEFAULT_DB_PATH,
) -> Optional[dict]:

    conn = get_connection(db_path)

    try:
        row = conn.execute(
            "SELECT * FROM cases WHERE id = ?",
            (case_id,),
        ).fetchone()

        if row is None:
            return None

        return _row_to_dict(row)

    finally:
        conn.close()


def update_review(
    case_id: int,
    human_urgency: str,
    human_notes: str = "",
    db_path: str = DEFAULT_DB_PATH,
) -> Optional[dict]:

    conn = get_connection(db_path)

    try:
        cursor = conn.execute(
            """
            UPDATE cases
            SET
                human_reviewed = 1,
                human_urgency = ?,
                human_notes = ?,
                reviewed_at = ?
            WHERE id = ?
            """,
            (
                human_urgency,
                human_notes,
                datetime.now(
                    timezone.utc
                ).isoformat(),
                case_id,
            ),
        )

        conn.commit()

        if cursor.rowcount == 0:
            return None

        row = conn.execute(
            "SELECT * FROM cases WHERE id = ?",
            (case_id,),
        ).fetchone()

        return _row_to_dict(row)

    finally:
        conn.close()


def clear_cases(
    db_path: str = DEFAULT_DB_PATH,
) -> int:

    conn = get_connection(db_path)

    try:
        cursor = conn.execute(
            "SELECT COUNT(*) FROM cases"
        )

        count = cursor.fetchone()[0]

        conn.execute(
            "DELETE FROM cases"
        )

        conn.commit()

        return count

    finally:
        conn.close()


def summarize_cases(
    cases: list[dict],
) -> dict:

    n = len(cases)

    if n == 0:
        return {
            "total_messages": 0,
            "by_urgency": {},
            "by_category": {},
            "escalated_count": 0,
            "escalation_rate": 0.0,
            "average_urgency_confidence": 0.0,
        }

    by_urgency = {}
    by_category = {}

    escalated_count = 0

    confidence_values = []

    for case in cases:

        urgency = case["urgency"]

        by_urgency[urgency] = (
            by_urgency.get(
                urgency,
                0,
            )
            + 1
        )

        category = case.get("category")

        if category:
            by_category[category] = (
                by_category.get(
                    category,
                    0,
                )
                + 1
            )

        if case["escalate_to_human"]:
            escalated_count += 1

        confidence = case.get(
            "urgency_confidence"
        )

        if confidence is not None:
            confidence_values.append(
                confidence
            )

    if confidence_values:
        average_confidence = (
            sum(confidence_values)
            / len(confidence_values)
        )
    else:
        average_confidence = 0.0

    return {
        "total_messages": n,
        "by_urgency": by_urgency,
        "by_category": by_category,
        "escalated_count": escalated_count,
        "escalation_rate": round(
            escalated_count / n,
            3,
        ),
        "average_urgency_confidence": round(
            average_confidence,
            3,
        ),
    }