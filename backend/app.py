"""
Run:
    python app.py
    (defaults to training from ./data/training_cases.csv on startup if
    that file exists, so the very first request already has a model.)
"""

import os
import tempfile

from flask import Flask, jsonify, request

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
    return jsonify(result.to_dict())


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
        summary = UrgencyAnalyzer.summarize(results)
        return jsonify({
            "summary": summary,
            "results": [r.to_dict() for r in results],
        })
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    finally:
        os.remove(tmp_path)


if __name__ == "__main__":
    _try_initial_training()
    app.run(host="0.0.0.0", port=5000, debug=True)