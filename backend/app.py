import os
import tempfile

from flask import Flask, jsonify, request

import database
from data_loader import load_training_data, load_inbox_csv
from urgency_analyzer import UrgencyAnalyzer

app = Flask(__name__)
analyzer = UrgencyAnalyzer()

DEFAULT_TRAINING_CSV = os.path.join(
    os.path.dirname(__file__), "data", "HumanFirst_AI_30_Test_Messages.csv"
)


def _try_initial_training():
    """Best-effort auto-train on startup if the default CSV is present."""
    if os.path.exists(DEFAULT_TRAINING_CSV):
        try:
            df = load_training_data(DEFAULT_TRAINING_CSV)
            info = analyzer.train(df)
            print(f"[startup] Trained on {DEFAULT_TRAINING_CSV}: {info}")
        except Exception as e:
            print(f"[startup] Could not auto-train: {e}")
    else:
        print(
            f"[startup] No training CSV found at {DEFAULT_TRAINING_CSV}. "
            "Using keyword-only scoring until /train is called."
        )


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model_trained": analyzer.is_trained,
    })


@app.route("/triage", methods=["POST"])
def triage():
    payload = request.get_json(silent=True) or {}
    message = payload.get("message", "").strip()

    if not message:
        return jsonify({"error": "Request body must include a non-empty 'message'."}), 400

    result = analyzer.analyze(message)
    stored = database.insert_case(result.to_dict(), source="single")
    return jsonify(stored)


@app.route("/train", methods=["POST"])
def train():
    if "file" not in request.files:
        return jsonify({"error": "Upload a CSV file under the 'file' field."}), 400

    upload = request.files["file"]
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        upload.save(tmp.name)
        tmp_path = tmp.name

    try:
        df = load_training_data(tmp_path)
        info = analyzer.train(df)
        return jsonify(info)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    finally:
        os.remove(tmp_path)


@app.route("/analyze", methods=["POST"])
def analyze():
    if "file" not in request.files:
        return jsonify({"error": "Upload a CSV file under the 'file' field."}), 400

    upload = request.files["file"]
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        upload.save(tmp.name)
        tmp_path = tmp.name

    try:
        df = load_inbox_csv(tmp_path)
        results = analyzer.analyze_batch(df["message"])
        stored = database.insert_cases_batch([r.to_dict() for r in results], source="batch")
        summary = database.summarize_cases(stored)
        return jsonify({
            "summary": summary,
            "results": stored,
        })
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    finally:
        os.remove(tmp_path)


def _parse_escalate_param():
    raw = request.args.get("escalate_to_human")
    if raw is None:
        return None
    return raw.strip().lower() in {"1", "true", "yes"}


@app.route("/cases", methods=["GET"])
def list_cases():
    limit = request.args.get("limit", type=int)
    cases = database.get_cases(
        urgency=request.args.get("urgency"),
        category=request.args.get("category"),
        escalate_to_human=_parse_escalate_param(),
        limit=limit,
    )
    return jsonify({"cases": cases, "count": len(cases)})


@app.route("/cases/summary", methods=["GET"])
def cases_summary():
    cases = database.get_cases(
        urgency=request.args.get("urgency"),
        category=request.args.get("category"),
        escalate_to_human=_parse_escalate_param(),
    )
    return jsonify(database.summarize_cases(cases))


@app.route("/cases/<int:case_id>/review", methods=["POST"])
def review_case(case_id):
    payload = request.get_json(silent=True) or {}
    human_urgency = payload.get("human_urgency", "").strip()
    human_notes = payload.get("human_notes", "").strip()

    if human_urgency not in {"Critical", "High", "Normal"}:
        return jsonify({
            "error": "human_urgency must be one of 'Critical', 'High', 'Normal'."
        }), 400

    updated = database.update_review(case_id, human_urgency, human_notes)
    if updated is None:
        return jsonify({"error": f"No case with id {case_id}."}), 404
    return jsonify(updated)


@app.route("/cases", methods=["DELETE"])
def delete_all_cases():
    removed = database.clear_cases()
    return jsonify({"deleted": removed})


if __name__ == "__main__":
    database.init_db()
    _try_initial_training()
    app.run(host="0.0.0.0", port=5000, debug=True)