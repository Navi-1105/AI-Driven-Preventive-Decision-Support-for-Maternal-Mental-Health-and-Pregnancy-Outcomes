import argparse
import hashlib
import json
import random
from importlib import metadata
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

try:
    from imblearn.over_sampling import SMOTE
except Exception:
    SMOTE = None

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
ROOT_DIR = Path(__file__).resolve().parents[3]
BACKEND_DIR = Path(__file__).resolve().parents[1]


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


def _random_oversample(X_train: pd.DataFrame, y_train: np.ndarray) -> tuple[pd.DataFrame, np.ndarray]:
    y_series = pd.Series(y_train)
    counts = y_series.value_counts()
    if len(counts) < 2:
        return X_train, y_train
    maj = int(counts.max())
    rng = np.random.RandomState(RANDOM_STATE)
    sampled_idx: list[int] = []
    for cls, _ in counts.items():
        cls_idx = y_series[y_series == cls].index.to_numpy()
        sampled = rng.choice(cls_idx, size=maj, replace=True)
        sampled_idx.extend(sampled.tolist())
    rng.shuffle(sampled_idx)
    X_os = X_train.iloc[sampled_idx].reset_index(drop=True)
    y_os = y_series.iloc[sampled_idx].to_numpy()
    return X_os, y_os


def _balance_train(
    X_train: pd.DataFrame, y_train: np.ndarray, strategy: str
) -> tuple[pd.DataFrame, np.ndarray, str]:
    if strategy == "none":
        return X_train, y_train, "none"
    if strategy == "smote":
        if SMOTE is None:
            X_os, y_os = _random_oversample(X_train, y_train)
            return X_os, y_os, "random_fallback"
        smote = SMOTE(random_state=RANDOM_STATE)
        X_sm, y_sm = smote.fit_resample(X_train, y_train)
        if not isinstance(X_sm, pd.DataFrame):
            X_sm = pd.DataFrame(X_sm, columns=X_train.columns)
        return X_sm, y_sm, "smote"
    X_os, y_os = _random_oversample(X_train, y_train)
    return X_os, y_os, "random"


def _build_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "xgb",
                XGBClassifier(
                    n_estimators=300,
                    max_depth=4,
                    learning_rate=0.05,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    min_child_weight=3,
                    reg_lambda=2.0,
                    random_state=RANDOM_STATE,
                    eval_metric="logloss",
                ),
            ),
        ]
    )


def _coerce_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    money_col = "Sufficient Money for Basic Needs"
    if money_col in out.columns:
        raw = out[money_col].astype(str).str.strip().str.lower()
        mapped = raw.map({"yes": 1, "no": 0})
        numeric_money = pd.to_numeric(out[money_col], errors="coerce")
        out.loc[:, money_col] = mapped.where(mapped.notna(), numeric_money)
    for col in DEFAULT_FEATURES:
        out.loc[:, col] = pd.to_numeric(out[col], errors="coerce")
    return out


def train(
    csv_path: Path,
    model_path: Path,
    metrics_path: Path | None = None,
    balance_strategy: str = "smote",
    cv_folds: int = 5,
):
    random.seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)

    df = pd.read_csv(csv_path)
    df = df[DEFAULT_FEATURES + [TARGET]]
    df[DEFAULT_FEATURES] = _coerce_feature_frame(df[DEFAULT_FEATURES])
    df = df.dropna(subset=DEFAULT_FEATURES + [TARGET])

    X = df[DEFAULT_FEATURES]
    y_raw = df[TARGET].astype(str).str.strip()

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    X_train_bal, y_train_bal, applied_balance = _balance_train(X_train, y_train, balance_strategy)

    pipeline = _build_pipeline()

    pipeline.fit(X_train_bal, y_train_bal)
    y_pred = pipeline.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    y_proba = pipeline.predict_proba(X_test)
    y_pos = y_proba[:, 1] if y_proba.shape[1] > 1 else y_proba[:, 0]

    cv_summary = {}
    if cv_folds >= 2:
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
        cv_pipe = _build_pipeline()
        cv_scores = cross_validate(
            cv_pipe,
            X,
            y,
            cv=cv,
            scoring={"roc_auc": "roc_auc", "f1_weighted": "f1_weighted"},
            n_jobs=None,
        )
        cv_summary = {
            "folds": cv_folds,
            "roc_auc_mean": round(float(np.mean(cv_scores["test_roc_auc"])), 4),
            "roc_auc_std": round(float(np.std(cv_scores["test_roc_auc"])), 4),
            "f1_weighted_mean": round(float(np.mean(cv_scores["test_f1_weighted"])), 4),
            "f1_weighted_std": round(float(np.std(cv_scores["test_f1_weighted"])), 4),
        }

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
            "class_balance_before_train": {str(k): int(v) for k, v in pd.Series(y_train).value_counts().to_dict().items()},
            "class_balance_after_train_balancing": {str(k): int(v) for k, v in pd.Series(y_train_bal).value_counts().to_dict().items()},
            "balance_strategy_requested": balance_strategy,
            "balance_strategy_applied": applied_balance,
            "dataset_sha256": _sha256sum(csv_path),
            "library_versions": {
                "sklearn": _pkg_version("scikit-learn"),
                "xgboost": _pkg_version("xgboost"),
                "shap": _pkg_version("shap"),
                "imblearn": _pkg_version("imbalanced-learn"),
            },
        },
    }
    if len(label_encoder.classes_) == 2:
        metrics["roc_auc"] = round(float(roc_auc_score(y_test, y_pos)), 4)
        metrics["pr_auc"] = round(float(average_precision_score(y_test, y_pos)), 4)
        metrics["brier_score"] = round(float(brier_score_loss(y_test, y_pos)), 4)
    pr, rc, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="weighted")
    metrics["precision_weighted"] = round(float(pr), 4)
    metrics["recall_weighted"] = round(float(rc), 4)
    metrics["f1_weighted"] = round(float(f1), 4)
    if cv_summary:
        metrics["cross_validation"] = cv_summary

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
        default=ROOT_DIR / "dataset.csv",
        help="Path to training CSV",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=BACKEND_DIR / "model/risk_model.joblib",
        help="Output model file",
    )
    parser.add_argument(
        "--metrics-out",
        type=Path,
        default=BACKEND_DIR / "model/metrics/risk_metrics.json",
        help="Output metrics json",
    )
    parser.add_argument(
        "--balance-strategy",
        type=str,
        default="smote",
        choices=["smote", "random", "none"],
        help="Class balancing strategy applied to training split only",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of stratified CV folds for audit metrics (set <2 to disable)",
    )
    args = parser.parse_args()

    score = train(
        args.csv,
        args.out,
        args.metrics_out,
        balance_strategy=args.balance_strategy,
        cv_folds=args.cv_folds,
    )
    print(f"Model trained. Accuracy: {score:.3f}. Saved to {args.out}")


if __name__ == "__main__":
    main()
