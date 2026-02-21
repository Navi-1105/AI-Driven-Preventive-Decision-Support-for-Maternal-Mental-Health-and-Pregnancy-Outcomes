import argparse
import hashlib
import json
import random
from importlib import metadata
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
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
RANDOM_STATE = 42
TEST_SIZE = 0.2


def _sha256sum(file_path: Path) -> str:
    h = hashlib.sha256()
    with file_path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _pkg_version(pkg_name: str) -> str:
    try:
        return metadata.version(pkg_name)
    except metadata.PackageNotFoundError:
        return "not-installed"


def train(csv_path: Path, model_path: Path, metrics_path: Path | None = None):
    random.seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)

    df = pd.read_csv(csv_path)
    df = df[DEFAULT_FEATURES + [TARGET]].dropna()

    X = df[DEFAULT_FEATURES]
    y_raw = df[TARGET].astype(str).str.strip()

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "xgb",
                XGBClassifier(
                    n_estimators=300,
                    random_state=RANDOM_STATE,
                    eval_metric="logloss",
                ),
            ),
        ]
    )

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    y_proba = pipeline.predict_proba(X_test)

    metrics = {
        "accuracy": round(float(score), 4),
        "classification_report": classification_report(
            y_test, y_pred, target_names=list(label_encoder.classes_), output_dict=True
        ),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "reproducibility": {
            "random_state": RANDOM_STATE,
            "split_strategy": "stratified",
            "test_size": TEST_SIZE,
            "train_size": 1 - TEST_SIZE,
            "feature_order": DEFAULT_FEATURES,
            "oversampling": (
                "SMOTE on training split only in experiment pipeline; "
                "this script consumes a preprocessed CSV"
            ),
            "dataset_sha256": _sha256sum(csv_path),
            "library_versions": {
                "sklearn": _pkg_version("scikit-learn"),
                "xgboost": _pkg_version("xgboost"),
                "shap": _pkg_version("shap"),
            },
        },
    }
    if len(label_encoder.classes_) == 2:
        metrics["roc_auc"] = round(float(roc_auc_score(y_test, y_proba[:, 1])), 4)
    pr, rc, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="weighted")
    metrics["precision_weighted"] = round(float(pr), 4)
    metrics["recall_weighted"] = round(float(rc), 4)
    metrics["f1_weighted"] = round(float(f1), 4)

    model_payload = {
        "pipeline": pipeline,
        "features": DEFAULT_FEATURES,
        "label_classes": list(label_encoder.classes_),
        "target": TARGET,
        "accuracy": score,
        "metrics": metrics,
        "reproducibility": metrics["reproducibility"],
    }

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_payload, model_path)
    if metrics_path is not None:
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("../oversampled_data.csv"),
        help="Path to training CSV",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("../model/risk_model.joblib"),
        help="Output model file",
    )
    parser.add_argument(
        "--metrics-out",
        type=Path,
        default=Path("../model/metrics/risk_metrics.json"),
        help="Output metrics json",
    )
    args = parser.parse_args()

    score = train(args.csv, args.out, args.metrics_out)
    print(f"Model trained. Accuracy: {score:.3f}. Saved to {args.out}")


if __name__ == "__main__":
    main()
