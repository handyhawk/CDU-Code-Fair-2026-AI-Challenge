import os
import tempfile

from flask import Flask, jsonify, request

import database

from data_loader import (
    load_training_data,
    load_inbox_csv,
)

from urgency_analyzer import UrgencyAnalyzer

from openai_analyzer import (
    classify_message,
    to_backend_dict,
)


app = Flask(__name__)


# ---------------------------------------------------------------------
# Local ML fallback
# ---------------------------------------------------------------------

analyzer = UrgencyAnalyzer()


DEFAULT_TRAINING_CSV = os.path.join(
    os.path.dirname(__file__),
    "data",
    "HumanFirst_AI_30_Test_Messages.csv",
)


def _try_initial_training():
    """
    Train the local ML model when the backend starts.

    This model is NOT the primary classifier anymore.

    Its job is to act as the fallback if OpenAI is unavailable.
    """

    if os.path.exists(
        DEFAULT_TRAINING_CSV
    ):
        try:
            df = load_training_data(
                DEFAULT_TRAINING_CSV
            )

            info = analyzer.train(df)

            print(
                "[startup] Local fallback "
                f"trained: {info}"
            )

        except Exception as exc:
            print(
                "[startup] Could not train "
                f"local fallback: {exc}"
            )

    else:
        print(
            "[startup] No local training "
            "CSV found. Local fallback "
            "will use keyword-only logic."
        )


# ---------------------------------------------------------------------
# Hybrid AI classification
# ---------------------------------------------------------------------

def analyse_with_fallback(
    message: str,
) -> dict:
    """
    HumanFirst hybrid AI flow.

    1. Try OpenAI primary classifier.
    2. If OpenAI fails, use local ML.
    3. Local ML results are always provisional.
    4. Local ML results always require human review.
    """

    try:
        openai_result = classify_message(
            message
        )

        result = to_backend_dict(
            message,
            openai_result,
        )

        print(
            "[analysis] OpenAI primary "
            "classifier used."
        )

        return result

    except Exception as exc:

        print(
            "[OpenAI unavailable] "
            f"{type(exc).__name__}: {exc}"
        )

        # -------------------------------------------------------------
        # Local ML fallback
        # -------------------------------------------------------------

        local_result = analyzer.analyze(
            message
        ).to_dict()

        local_result[
            "analysis_mode"
        ] = "Local ML"

        local_result[
            "provisional"
        ] = True

        # A fallback result must ALWAYS
        # be reviewed by a human.
        local_result[
            "escalate_to_human"
        ] = True

        existing_explanation = (
            local_result.get(
                "explanation",
                "",
            )
        )

        local_result["explanation"] = (
            existing_explanation
            + " OpenAI was unavailable, "
              "so the local fallback model "
              "was used. This result is "
              "provisional and requires "
              "human review."
        ).strip()

        return local_result


# ---------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------

@app.route(
    "/health",
    methods=["GET"],
)
def health():

    return jsonify(
        {
            "status": "ok",
            "local_model_trained":
                analyzer.is_trained,
            "primary_model":
                "OpenAI",
            "fallback_model":
                "Local ML",
        }
    )


# ---------------------------------------------------------------------
# Single-message triage
# ---------------------------------------------------------------------

@app.route(
    "/triage",
    methods=["POST"],
)
def triage():

    payload = (
        request.get_json(
            silent=True
        )
        or {}
    )

    message = (
        payload.get(
            "message",
            "",
        )
        .strip()
    )

    if not message:
        return jsonify(
            {
                "error":
                    "Request body must "
                    "include a non-empty "
                    "'message'."
            }
        ), 400

    result = analyse_with_fallback(
        message
    )

    stored = database.insert_case(
        result,
        source="single",
    )

    return jsonify(stored)


# ---------------------------------------------------------------------
# Train / retrain local fallback
# ---------------------------------------------------------------------

@app.route(
    "/train",
    methods=["POST"],
)
def train():

    if "file" not in request.files:
        return jsonify(
            {
                "error":
                    "Upload a CSV file "
                    "under the 'file' field."
            }
        ), 400

    upload = request.files["file"]

    with tempfile.NamedTemporaryFile(
        suffix=".csv",
        delete=False,
    ) as tmp:

        upload.save(
            tmp.name
        )

        tmp_path = tmp.name

    try:
        df = load_training_data(
            tmp_path
        )

        info = analyzer.train(df)

        return jsonify(info)

    except ValueError as exc:

        return jsonify(
            {
                "error": str(exc)
            }
        ), 400

    finally:
        os.remove(tmp_path)


# ---------------------------------------------------------------------
# Batch analysis
# ---------------------------------------------------------------------

@app.route(
    "/analyze",
    methods=["POST"],
)
def analyze():

    if "file" not in request.files:
        return jsonify(
            {
                "error":
                    "Upload a CSV file "
                    "under the 'file' field."
            }
        ), 400

    upload = request.files["file"]

    with tempfile.NamedTemporaryFile(
        suffix=".csv",
        delete=False,
    ) as tmp:

        upload.save(
            tmp.name
        )

        tmp_path = tmp.name

    try:
        df = load_inbox_csv(
            tmp_path
        )

        analyses = []

        for message in df["message"]:
            result = analyse_with_fallback(
                message
            )

            analyses.append(result)

        stored = (
            database.insert_cases_batch(
                analyses,
                source="batch",
            )
        )

        summary = (
            database.summarize_cases(
                stored
            )
        )

        return jsonify(
            {
                "summary": summary,
                "results": stored,
            }
        )

    except ValueError as exc:

        return jsonify(
            {
                "error": str(exc)
            }
        ), 400

    finally:
        os.remove(tmp_path)


# ---------------------------------------------------------------------
# Case filtering helper
# ---------------------------------------------------------------------

def _parse_escalate_param():

    raw = request.args.get(
        "escalate_to_human"
    )

    if raw is None:
        return None

    return (
        raw.strip().lower()
        in {
            "1",
            "true",
            "yes",
        }
    )


# ---------------------------------------------------------------------
# Stored case queue
# ---------------------------------------------------------------------

@app.route(
    "/cases",
    methods=["GET"],
)
def list_cases():

    limit = request.args.get(
        "limit",
        type=int,
    )

    cases = database.get_cases(
        urgency=request.args.get(
            "urgency"
        ),
        category=request.args.get(
            "category"
        ),
        escalate_to_human=(
            _parse_escalate_param()
        ),
        limit=limit,
    )

    return jsonify(
        {
            "cases": cases,
            "count": len(cases),
        }
    )


# ---------------------------------------------------------------------
# Case summary
# ---------------------------------------------------------------------

@app.route(
    "/cases/summary",
    methods=["GET"],
)
def cases_summary():

    cases = database.get_cases(
        urgency=request.args.get(
            "urgency"
        ),
        category=request.args.get(
            "category"
        ),
        escalate_to_human=(
            _parse_escalate_param()
        ),
    )

    return jsonify(
        database.summarize_cases(
            cases
        )
    )


# ---------------------------------------------------------------------
# Human review
# ---------------------------------------------------------------------

@app.route(
    "/cases/<int:case_id>/review",
    methods=["POST"],
)
def review_case(case_id):

    payload = (
        request.get_json(
            silent=True
        )
        or {}
    )

    human_urgency = (
        payload.get(
            "human_urgency",
            "",
        )
        .strip()
    )

    human_notes = (
        payload.get(
            "human_notes",
            "",
        )
        .strip()
    )

    if human_urgency not in {
        "Critical",
        "High",
        "Normal",
    }:

        return jsonify(
            {
                "error":
                    "human_urgency must "
                    "be one of 'Critical', "
                    "'High', or 'Normal'."
            }
        ), 400

    updated = database.update_review(
        case_id,
        human_urgency,
        human_notes,
    )

    if updated is None:

        return jsonify(
            {
                "error":
                    f"No case with id "
                    f"{case_id}."
            }
        ), 404

    return jsonify(updated)


# ---------------------------------------------------------------------
# Clear stored cases
# ---------------------------------------------------------------------

@app.route(
    "/cases",
    methods=["DELETE"],
)
def delete_all_cases():

    removed = (
        database.clear_cases()
    )

    return jsonify(
        {
            "deleted": removed
        }
    )


# ---------------------------------------------------------------------
# Start backend
# ---------------------------------------------------------------------

if __name__ == "__main__":

    database.init_db()

    _try_initial_training()

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True,
    )