import argparse
import json
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
from pymongo import MongoClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

DEFAULT_FEATURES = [
    "Age",
    "Gestational Age",
    "Trouble falling or staying sleep or sleeping too much",
    "Poor appetite or overeating",
    "Feeling tired or having little energy",
    "Sufficient Money for Basic Needs",
    "Thoughts that you would be better off dead, or of hurting yourself",
]
TARGET = "Labelling"

LABEL_MAP = {
    "high": "Depressed",
    "low": "Not",
    "agree": None,
    "medium": None,
}


def _map_prediction_to_row(prediction: dict, clinician_label: str) -> dict | None:
    target = LABEL_MAP.get(clinician_label.lower())
    if target is None:
        return None
    beh = prediction.get("behavioral", {})
    demo = prediction.get("demographics", {})
    return {
        "Age": float(demo.get("age", 0)),
        "Gestational Age": float(prediction.get("gestational_weeks", 0)),
        "Trouble falling or staying sleep or sleeping too much": float(
            round((10.0 - float(beh.get("sleep_quality", 5))) / 10.0 * 3)
        ),
        "Poor appetite or overeating": float(
            round((10.0 - float(beh.get("appetite", 5))) / 10.0 * 3)
        ),
        "Feeling tired or having little energy": float(
            round(float(beh.get("fatigue", 5)) / 10.0 * 3)
        ),
        "Sufficient Money for Basic Needs": float(
            round((10.0 - float(beh.get("financial_stress", 5))) / 10.0 * 3)
        ),
        "Thoughts that you would be better off dead, or of hurting yourself": float(
            prediction.get("self_harm", 0)
        ),
        TARGET: target,
    }


def build_feedback_frame(mongo_uri: str, mongo_db: str) -> pd.DataFrame:
    client = MongoClient(mongo_uri)
    db = client[mongo_db]

    predictions = {
        item.get("prediction_id"): item
        for item in db.predictions.find({}, {"_id": 0})
        if item.get("prediction_id")
    }

    rows = []
    for feedback in db.feedback.find({}, {"_id": 0}):
        prediction_id = feedback.get("prediction_id")
        if not prediction_id or prediction_id not in predictions:
            continue
        row = _map_prediction_to_row(predictions[prediction_id], feedback.get("clinician_label", ""))
        if row is not None:
            rows.append(row)

    return pd.DataFrame(rows)


def retrain(base_csv: Path, feedback_df: pd.DataFrame, model_dir: Path):
    base_df = pd.read_csv(base_csv)[DEFAULT_FEATURES + [TARGET]].dropna()

    combined = pd.concat([base_df, feedback_df], ignore_index=True) if not feedback_df.empty else base_df

    X = combined[DEFAULT_FEATURES]
    y_raw = combined[TARGET].astype(str).str.strip()

    encoder = LabelEncoder()
    y = encoder.fit_transform(y_raw)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("rf", RandomForestClassifier(n_estimators=300, random_state=42)),
        ]
    )
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    metrics = {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "classification_report": classification_report(
            y_test, y_pred, target_names=list(encoder.classes_), output_dict=True
        ),
        "feedback_rows_used": int(feedback_df.shape[0]),
    }

    version = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    version_path = model_dir / f"risk_model_v{version}.joblib"
    latest_path = model_dir / "risk_model.joblib"

    payload = {
        "pipeline": pipeline,
        "features": DEFAULT_FEATURES,
        "label_classes": list(encoder.classes_),
        "target": TARGET,
        "accuracy": metrics["accuracy"],
        "metrics": metrics,
        "version": version,
    }

    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "metrics").mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, version_path)
    joblib.dump(payload, latest_path)

    metrics_path = model_dir / "metrics" / f"risk_retrain_{version}.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return version, metrics_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=Path("../oversampled_data.csv"))
    parser.add_argument("--model-dir", type=Path, default=Path("model"))
    parser.add_argument("--mongo-uri", type=str, default="mongodb://localhost:27017")
    parser.add_argument("--mongo-db", type=str, default="perinatal_ai")
    parser.add_argument("--min-feedback", type=int, default=50)
    args = parser.parse_args()

    feedback_df = build_feedback_frame(args.mongo_uri, args.mongo_db)
    if feedback_df.shape[0] < args.min_feedback:
        print(
            f"Not enough feedback rows: {feedback_df.shape[0]} < {args.min_feedback}. Skipping retrain."
        )
        return

    version, metrics_path = retrain(args.csv, feedback_df, args.model_dir)
    print(f"Retrained model version {version}. Metrics written to {metrics_path}")


if __name__ == "__main__":
    main()
