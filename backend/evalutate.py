"""
Run:
    python evaluate.py [path/to/training_csv]
    (defaults to data/HumanFirst_AI_30_Test_Messages.csv)
"""

import sys
import os

import pandas as pd
from sklearn.model_selection import LeaveOneOut

from data_loader import load_training_data
from urgency_analyzer import UrgencyAnalyzer, _make_pipeline


def evaluate_field(df: pd.DataFrame, field: str) -> dict:
    if df[field].nunique() < 2:
        return {"field": field, "skipped": True, "reason": "only one class present"}

    loo = LeaveOneOut()
    correct = 0
    total = 0
    confusions = []
    correct_confidences = []
    incorrect_confidences = []

    X = df["message"].tolist()
    y = df[field].tolist()

    for train_idx, test_idx in loo.split(X):
        train_texts = [X[i] for i in train_idx]
        train_labels = [y[i] for i in train_idx]
        test_text = X[test_idx[0]]
        true_label = y[test_idx[0]]

        # Skip folds where the held-out row is the only example of its
        # class (can't be predicted correctly by definition, and skews
        # the estimate if included).
        if train_labels.count(true_label) == 0:
            continue

        pipeline = _make_pipeline()
        pipeline.fit(train_texts, train_labels)
        proba = pipeline.predict_proba([test_text])[0]
        best_idx = proba.argmax()
        pred_label = pipeline.classes_[best_idx]
        pred_confidence = float(proba[best_idx])

        total += 1
        if pred_label == true_label:
            correct += 1
            correct_confidences.append(pred_confidence)
        else:
            confusions.append((test_text[:60], true_label, pred_label))
            incorrect_confidences.append(pred_confidence)

    accuracy = correct / total if total else 0.0
    avg_correct_conf = (
        sum(correct_confidences) / len(correct_confidences) if correct_confidences else None
    )
    avg_incorrect_conf = (
        sum(incorrect_confidences) / len(incorrect_confidences) if incorrect_confidences else None
    )
    return {
        "field": field,
        "skipped": False,
        "n_evaluated": total,
        "accuracy": round(accuracy, 3),
        "avg_confidence_when_correct": round(avg_correct_conf, 3) if avg_correct_conf else None,
        "avg_confidence_when_incorrect": round(avg_incorrect_conf, 3) if avg_incorrect_conf else None,
        "misclassifications": confusions,
    }


def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(__file__), "data", "HumanFirst_AI_30_Test_Messages.csv"
    )
    df = load_training_data(csv_path)
    print(f"Loaded {len(df)} labeled examples from {csv_path}\n")

    for field in ["urgency", "category", "route"]:
        result = evaluate_field(df, field)
        print(f"--- {field} (leave-one-out) ---")
        if result["skipped"]:
            print(f"  skipped: {result['reason']}\n")
            continue
        print(f"  accuracy: {result['accuracy']:.0%}  ({result['n_evaluated']} folds evaluated)")
        if result["avg_confidence_when_correct"] is not None:
            print(f"  avg confidence when correct:   {result['avg_confidence_when_correct']:.0%}")
        if result["avg_confidence_when_incorrect"] is not None:
            print(f"  avg confidence when incorrect: {result['avg_confidence_when_incorrect']:.0%}")
        if result["misclassifications"]:
            print("  misclassified:")
            for text, true_label, pred_label in result["misclassifications"]:
                print(f"    \"{text}...\"  true={true_label}  predicted={pred_label}")
        print()


if __name__ == "__main__":
    main()