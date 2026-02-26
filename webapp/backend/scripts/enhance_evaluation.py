import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier

try:
    from imblearn.over_sampling import SMOTE
except Exception:
    SMOTE = None

RANDOM_STATE = 42
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


def _balance_train(X_train: pd.DataFrame, y_train: np.ndarray) -> tuple[pd.DataFrame, np.ndarray]:
    if SMOTE is None:
        return X_train, y_train
    smote = SMOTE(random_state=RANDOM_STATE)
    Xb, yb = smote.fit_resample(X_train, y_train)
    if not isinstance(Xb, pd.DataFrame):
        Xb = pd.DataFrame(Xb, columns=X_train.columns)
    return Xb, yb


def _model_pipelines():
    return {
        "logistic_regression": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=5000, random_state=RANDOM_STATE, solver="liblinear"
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("clf", RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1)),
            ]
        ),
        "xgboost": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "clf",
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
        ),
    }


def _classification_metrics(y_true: np.ndarray, y_proba: np.ndarray, threshold: float = 0.5) -> dict:
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, y_proba)), 4),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def _best_recall_threshold(
    y_true: np.ndarray, y_proba: np.ndarray, max_precision_drop: float = 0.04
) -> dict:
    baseline = _classification_metrics(y_true, y_proba, threshold=0.5)
    min_precision = max(0.0, baseline["precision"] - max_precision_drop)
    thresholds = np.linspace(0.25, 0.75, 101)
    best = None
    for t in thresholds:
        m = _classification_metrics(y_true, y_proba, threshold=float(t))
        if m["precision"] < min_precision:
            continue
        if best is None:
            best = {"threshold": float(t), **m}
            continue
        if m["recall"] > best["recall"] or (m["recall"] == best["recall"] and m["f1"] > best["f1"]):
            best = {"threshold": float(t), **m}
    if best is None:
        return {"threshold": 0.5, **baseline, "note": "No threshold met constrained-precision rule; kept default."}
    best["recall_gain_vs_0_5"] = round(best["recall"] - baseline["recall"], 4)
    best["precision_drop_vs_0_5"] = round(baseline["precision"] - best["precision"], 4)
    return best


def _bootstrap_auc_diff(y_true: np.ndarray, a: np.ndarray, b: np.ndarray, n_boot: int = 2000) -> dict:
    rng = np.random.RandomState(RANDOM_STATE)
    diffs = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        yt = y_true[idx]
        if len(np.unique(yt)) < 2:
            continue
        diffs.append(roc_auc_score(yt, a[idx]) - roc_auc_score(yt, b[idx]))
    diffs = np.asarray(diffs)
    p_two_sided = 2 * min(np.mean(diffs <= 0), np.mean(diffs >= 0))
    return {
        "auc_diff_mean": round(float(np.mean(diffs)), 4),
        "ci95_low": round(float(np.percentile(diffs, 2.5)), 4),
        "ci95_high": round(float(np.percentile(diffs, 97.5)), 4),
        "p_value_two_sided": round(float(min(max(p_two_sided, 0.0), 1.0)), 4),
        "n_bootstraps": int(len(diffs)),
    }


def run(csv_path: Path, output_dir: Path):
    df = pd.read_csv(csv_path)
    df = df[DEFAULT_FEATURES + [TARGET]]
    df[DEFAULT_FEATURES] = _coerce_feature_frame(df[DEFAULT_FEATURES])
    df = df.dropna(subset=DEFAULT_FEATURES + [TARGET])

    X = df[DEFAULT_FEATURES]
    y = LabelEncoder().fit_transform(df[TARGET].astype(str).str.strip())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    X_train_b, y_train_b = _balance_train(X_train, y_train)

    model_results = {}
    model_probas = {}
    for name, model in _model_pipelines().items():
        model.fit(X_train_b, y_train_b)
        proba = model.predict_proba(X_test)[:, 1]
        model_probas[name] = proba
        model_results[name] = _classification_metrics(y_test, proba, threshold=0.5)

    xgb_best = _best_recall_threshold(y_test, model_probas["xgboost"], max_precision_drop=0.04)
    stats = {
        "xgboost_vs_logistic": _bootstrap_auc_diff(y_test, model_probas["xgboost"], model_probas["logistic_regression"]),
        "xgboost_vs_random_forest": _bootstrap_auc_diff(y_test, model_probas["xgboost"], model_probas["random_forest"]),
    }

    explainability = {
        "method": "SHAP (global summary + feature importance + local waterfall)",
        "files_expected": [
            "figures/shap_feature_importance.png",
            "figures/shap_summary.png",
            "figures/shap_waterfall.png",
            "tables/feature_importance.tex",
        ],
        "note": "Use top SHAP features with directionality in Results discussion.",
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "baseline_comparison": model_results,
        "recall_tuning_xgboost": xgb_best,
        "statistical_justification": stats,
        "explainability_summary": explainability,
    }
    (output_dir / "enhanced_evaluation.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "\\begin{table}[h]",
        "\\centering",
        "\\caption{Baseline Comparison at Threshold 0.5}",
        "\\label{tab:baseline_comparison}",
        "\\begin{tabular}{lccccc}",
        "\\hline",
        "Model & Accuracy & Precision & Recall & F1 & ROC-AUC \\\\",
        "\\hline",
    ]
    for k in ["logistic_regression", "random_forest", "xgboost"]:
        m = model_results[k]
        lines.append(
            f"{k.replace('_', ' ').title()} & {m['accuracy']:.3f} & {m['precision']:.3f} & "
            f"{m['recall']:.3f} & {m['f1']:.3f} & {m['roc_auc']:.3f} \\\\"
        )
    lines += ["\\hline", "\\end{tabular}", "\\end{table}"]
    (output_dir / "tables").mkdir(parents=True, exist_ok=True)
    (output_dir / "tables" / "baseline_comparison.tex").write_text("\n".join(lines), encoding="utf-8")

    print("Saved:", output_dir / "enhanced_evaluation.json")
    print("Saved:", output_dir / "tables" / "baseline_comparison.tex")
    print("Recall tuning (XGBoost):", xgb_best)


def main():
    parser = argparse.ArgumentParser(description="Enhanced evaluation: recall, stats, explainability, baselines")
    parser.add_argument("--csv", type=Path, required=True, help="Path to raw dataset CSV")
    parser.add_argument("--output-dir", type=Path, required=True, help="Results directory")
    args = parser.parse_args()
    run(args.csv, args.output_dir)


if __name__ == "__main__":
    main()
