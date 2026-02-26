import argparse
import json
import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
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
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

RANDOM_STATE = 42
TEST_SIZE = 0.2

ROOT_DIR = Path(__file__).resolve().parents[3]

COMMON_FEATURES = [
    "age",
    "gestational_weeks",
    "prior_pregnancy_loss",
    "support_score",
    "sleep_score",
    "fatigue_score",
    "low_mood_score",
    "suicidal_ideation_score",
    "economic_stress_score",
    "anxiety_score",
]

TARGET = "target"
SOURCE = "source"


def _as_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _to_yes_no_binary(series: pd.Series, yes_vals: set[str], no_vals: set[str]) -> pd.Series:
    s = series.astype(str).str.strip().str.lower()
    out = pd.Series(np.nan, index=series.index, dtype=float)
    out[s.isin(yes_vals)] = 1.0
    out[s.isin(no_vals)] = 0.0
    return out


def _parse_pregnancy_length_weeks(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip().str.lower()
    months = (
        s.str.extract(r"(\d+(?:\.\d+)?)", expand=False)
        .pipe(pd.to_numeric, errors="coerce")
    )
    weeks = months * 4.345
    return weeks.clip(lower=1, upper=42)


def _gestation_from_dates(eval_date: pd.Series, due_date: pd.Series) -> pd.Series:
    eval_dt = pd.to_datetime(eval_date, errors="coerce", dayfirst=True)
    due_dt = pd.to_datetime(due_date, errors="coerce", dayfirst=True)
    weeks_remaining = (due_dt - eval_dt).dt.days / 7.0
    gestation = 40.0 - weeks_remaining
    return gestation.clip(lower=1, upper=42)


def _build_main_frame(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    out = pd.DataFrame(index=df.index)
    out["age"] = _as_numeric(df["Age"])
    out["gestational_weeks"] = _as_numeric(df["Gestational Age"])
    out["prior_pregnancy_loss"] = _to_yes_no_binary(
        df["Previous Miscarriage"], {"yes"}, {"no"}
    )
    out["support_score"] = (
        df["Relationship with Mother in-law"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map({"poor": 0.0, "moderate": 1.0, "good": 2.0})
    )
    out["sleep_score"] = _as_numeric(
        df["Trouble falling or staying sleep or sleeping too much"]
    )
    out["fatigue_score"] = _as_numeric(df["Feeling tired or having little energy"])
    out["low_mood_score"] = _as_numeric(df["Feeling down, depressed, or hopeless"])
    out["suicidal_ideation_score"] = _as_numeric(
        df["Thoughts that you would be better off dead, or of hurting yourself"]
    )
    out["economic_stress_score"] = _to_yes_no_binary(
        df["Sufficient Money for Basic Needs"], {"no"}, {"yes"}
    )
    out["anxiety_score"] = np.nan

    target = (
        df["Labelling"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map({"depressed": 1, "not": 0})
    )
    out[TARGET] = target
    out[SOURCE] = "main"
    return out


def _build_ppd_frame(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    out = pd.DataFrame(index=df.index)
    out["age"] = _as_numeric(df["Age"])
    out["gestational_weeks"] = _parse_pregnancy_length_weeks(df["Pregnancy length"])
    out["prior_pregnancy_loss"] = (
        df["History of pregnancy loss"].astype(str).str.strip().ne("").astype(float)
    )
    out["support_score"] = (
        df["Recieved Support"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map({"low": 0.0, "medium": 1.0, "high": 2.0})
    )

    sleep_1 = _to_yes_no_binary(
        df["Relax/sleep when newborn is tended "], {"no"}, {"yes"}
    )
    sleep_2 = _to_yes_no_binary(
        df["Relax/sleep when the newborn is asleep"], {"no"}, {"yes"}
    )
    out["sleep_score"] = (sleep_1.fillna(0) + sleep_2.fillna(0)) * 1.5

    out["fatigue_score"] = (
        df["Feeling for regular activities"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map({"tired": 3.0, "worried": 2.0, "afraid": 2.0})
        .fillna(1.0)
    )
    out["low_mood_score"] = _to_yes_no_binary(
        df["Depression during pregnancy (PHQ2)"], {"positive"}, {"negative"}
    ) * 3.0
    out["suicidal_ideation_score"] = np.nan
    out["economic_stress_score"] = np.nan
    out["anxiety_score"] = np.nan

    epds_score = _as_numeric(df["EPDS Score"])
    phq9_score = _as_numeric(df["PHQ9 Score"])
    epds_result = df["EPDS Result"].astype(str).str.strip().str.lower()
    phq9_result = df["PHQ9 Result"].astype(str).str.strip().str.lower()
    target = (
        (epds_score >= 13)
        | (epds_result == "high")
        | (phq9_score >= 10)
        | (phq9_result.isin({"moderate", "moderately severe", "severe"}))
    ).astype(int)
    out[TARGET] = target
    out[SOURCE] = "ppd"
    return out


def _build_uganda_frame(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    out = pd.DataFrame(index=df.index)
    out["age"] = _as_numeric(df["age"])
    out["gestational_weeks"] = _gestation_from_dates(df["evaluation_date"], df["dueDate"])
    out["prior_pregnancy_loss"] = (
        _as_numeric(df["numStillBorn"]).fillna(0) + _as_numeric(df["numMissCarry"]).fillna(0)
    ).gt(0).astype(float)
    out["support_score"] = np.nan
    out["sleep_score"] = np.nan
    out["fatigue_score"] = np.nan
    out["low_mood_score"] = np.nan
    out["suicidal_ideation_score"] = np.nan
    out["economic_stress_score"] = np.nan
    out["anxiety_score"] = _as_numeric(df["Anxiety"])

    total_score = _as_numeric(df["Total Score"])
    out[TARGET] = (total_score >= 13).astype(int)
    out[SOURCE] = "uganda"
    return out


def _oversample_train(X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    counts = y.value_counts()
    if len(counts) < 2:
        return X, y
    maj_class = counts.idxmax()
    min_class = counts.idxmin()
    n_maj = int(counts.max())
    idx_maj = y[y == maj_class].index
    idx_min = y[y == min_class].index
    sampled_min = np.random.RandomState(RANDOM_STATE).choice(idx_min, size=n_maj, replace=True)
    idx_all = np.concatenate([idx_maj, sampled_min])
    X_os = X.loc[idx_all].sample(frac=1.0, random_state=RANDOM_STATE)
    y_os = y.loc[X_os.index]
    return X_os, y_os


def _assemble_dataset(main_csv: Path, ppd_csv: Path, uganda_csv: Path, include: set[str]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    if "main" in include:
        frames.append(_build_main_frame(main_csv))
    if "ppd" in include:
        frames.append(_build_ppd_frame(ppd_csv))
    if "uganda" in include:
        frames.append(_build_uganda_frame(uganda_csv))
    if not frames:
        raise ValueError("No dataset selected. Use --include with at least one dataset.")
    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=[TARGET])
    return df


def train(
    main_csv: Path,
    ppd_csv: Path,
    uganda_csv: Path,
    include: set[str],
    model_path: Path,
    metrics_path: Path | None = None,
    oversample_train: bool = True,
) -> float:
    random.seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)

    df = _assemble_dataset(main_csv, ppd_csv, uganda_csv, include)
    X = df[COMMON_FEATURES].copy()
    y = df[TARGET].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    if oversample_train:
        X_train, y_train = _oversample_train(X_train, y_train)

    pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "xgb",
                XGBClassifier(
                    n_estimators=250,
                    max_depth=4,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    min_child_weight=3,
                    reg_lambda=2.0,
                    random_state=RANDOM_STATE,
                    eval_metric="logloss",
                ),
            ),
        ]
    )

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    score = accuracy_score(y_test, y_pred)

    pr, rc, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="weighted")
    metrics = {
        "accuracy": round(float(score), 4),
        "roc_auc": round(float(roc_auc_score(y_test, y_proba)), 4),
        "precision_weighted": round(float(pr), 4),
        "recall_weighted": round(float(rc), 4),
        "f1_weighted": round(float(f1), 4),
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "reproducibility": {
            "random_state": RANDOM_STATE,
            "test_size": TEST_SIZE,
            "train_size": 1 - TEST_SIZE,
            "oversample_train_only": oversample_train,
            "features": COMMON_FEATURES,
            "included_datasets": sorted(include),
            "full_dataset_rows": int(len(df)),
            "class_balance_full": {str(k): int(v) for k, v in y.value_counts().to_dict().items()},
            "class_balance_train_after_oversample": {
                str(k): int(v) for k, v in y_train.value_counts().to_dict().items()
            },
            "source_counts": {
                str(k): int(v) for k, v in df[SOURCE].value_counts().to_dict().items()
            },
        },
    }

    model_payload = {
        "pipeline": pipeline,
        "features": COMMON_FEATURES,
        "label_classes": ["Not", "Depressed"],
        "target": TARGET,
        "accuracy": score,
        "metrics": metrics,
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
        "--main-csv",
        type=Path,
        default=ROOT_DIR / "dataset.csv",
        help="Path to the project main dataset.csv",
    )
    parser.add_argument(
        "--ppd-csv",
        type=Path,
        default=ROOT_DIR / "Data for Postpartum Depression Prediction in Bangladesh/PPD_dataset_v2.csv",
        help="Path to the Bangladesh PPD dataset",
    )
    parser.add_argument(
        "--uganda-csv",
        type=Path,
        default=ROOT_DIR / "Large Scale Anonymized EPDS Data for Prenatal Women in Selected hospitals in Uganda/records.csv",
        help="Path to the Uganda prenatal dataset",
    )
    parser.add_argument(
        "--include",
        type=str,
        default="main,ppd,uganda",
        help="Comma-separated list from: main,ppd,uganda",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("../model/risk_model_combined.joblib"),
        help="Output model file",
    )
    parser.add_argument(
        "--metrics-out",
        type=Path,
        default=Path("../model/metrics/risk_metrics_combined.json"),
        help="Output metrics json",
    )
    parser.add_argument(
        "--disable-oversample",
        action="store_true",
        help="Disable random oversampling on the training split",
    )
    args = parser.parse_args()

    include = {v.strip().lower() for v in args.include.split(",") if v.strip()}
    score = train(
        main_csv=args.main_csv,
        ppd_csv=args.ppd_csv,
        uganda_csv=args.uganda_csv,
        include=include,
        model_path=args.out,
        metrics_path=args.metrics_out,
        oversample_train=not args.disable_oversample,
    )
    print(f"Combined model trained. Accuracy: {score:.3f}. Saved to {args.out}")


if __name__ == "__main__":
    main()
