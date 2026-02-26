import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, learning_curve, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

RANDOM_STATE = 42
TARGET = "Labelling"
SCORE_COL = "Scalling"
SYMPTOM_FEATURES = [
    "Little interest or pleasure in doing things",
    "Feeling down, depressed, or hopeless",
    "Trouble falling or staying sleep or sleeping too much",
    "Feeling tired or having little energy",
    "Poor appetite or overeating",
    "Feeling badabout yourself that you are failure or have let yourself or your family down",
    "Trouble concentrating on things, such as reading the newspaper or watching television ",
    "Moving or speaking so slowly that other people could have Noticed. ",
    "Thoughts that you would be better off dead, or of hurting yourself",
]
BASELINE_FEATURES = [
    "Age",
    "Gestational Age",
    "Total Number of Children",
    "Gravida",
    "Female Education",
    "Husband Education",
    "Working Status",
    "Previous Miscarriage",
    "Sufficient Money for Basic Needs",
]

try:
    plt.style.use("seaborn-v0_8-paper")
except OSError:
    plt.style.use("seaborn-paper")
sns.set_palette("deep")


def _target_binary(s: pd.Series) -> np.ndarray:
    return s.astype(str).str.strip().str.lower().eq("depressed").astype(int).to_numpy()


def _build_pipeline(df_features: pd.DataFrame, y: np.ndarray) -> Pipeline:
    num_cols = [c for c in df_features.columns if pd.api.types.is_numeric_dtype(df_features[c])]
    cat_cols = [c for c in df_features.columns if c not in num_cols]

    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), num_cols),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                cat_cols,
            ),
        ]
    )

    pos = int(np.sum(y == 1))
    neg = int(np.sum(y == 0))
    scale_pos_weight = float(neg / pos) if pos > 0 else 1.0

    clf = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=3,
        reg_lambda=2.0,
        random_state=RANDOM_STATE,
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
    )
    return Pipeline([("preprocess", pre), ("xgb", clf)])


def _metrics(y_true: np.ndarray, y_proba: np.ndarray, threshold: float = 0.5) -> dict:
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, y_proba)), 4),
        "brier_score": round(float(brier_score_loss(y_true, y_proba)), 4),
    }


def _best_recall_threshold(y_true: np.ndarray, y_proba: np.ndarray, max_precision_drop: float = 0.04) -> dict:
    baseline = _metrics(y_true, y_proba, threshold=0.5)
    min_precision = max(0.0, baseline["precision"] - max_precision_drop)
    best = {"threshold": 0.5, **baseline}
    for t in np.linspace(0.25, 0.75, 101):
        m = _metrics(y_true, y_proba, threshold=float(t))
        if m["precision"] < min_precision:
            continue
        if m["recall"] > best["recall"] or (m["recall"] == best["recall"] and m["f1"] > best["f1"]):
            best = {"threshold": float(t), **m}
    best["recall_gain_vs_0_5"] = round(best["recall"] - baseline["recall"], 4)
    best["precision_drop_vs_0_5"] = round(baseline["precision"] - best["precision"], 4)
    return best


def _learning_curve_auc(pipeline: Pipeline, X: pd.DataFrame, y: np.ndarray) -> dict:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    train_sizes, train_scores, test_scores = learning_curve(
        pipeline,
        X,
        y,
        cv=cv,
        train_sizes=np.linspace(0.2, 1.0, 5),
        scoring="roc_auc",
        n_jobs=None,
    )
    return {
        "train_sizes": train_sizes.tolist(),
        "train_auc_mean": np.mean(train_scores, axis=1).round(4).tolist(),
        "train_auc_std": np.std(train_scores, axis=1).round(4).tolist(),
        "cv_auc_mean": np.mean(test_scores, axis=1).round(4).tolist(),
        "cv_auc_std": np.std(test_scores, axis=1).round(4).tolist(),
    }


def _plot_bar(summary: dict, metric: str, out: Path):
    names = list(summary.keys())
    vals = [summary[n]["test_default_threshold"][metric] for n in names]
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x=names, y=vals, ax=ax)
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_xlabel("Model")
    ax.set_title(f"{metric.replace('_', ' ').title()} Comparison")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.005, f"{v:.3f}", ha="center", fontsize=9)
    plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()


def _plot_learning_curve(lc: dict, out: Path):
    x = lc["train_sizes"]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, lc["train_auc_mean"], marker="o", label="Train AUC")
    ax.plot(x, lc["cv_auc_mean"], marker="o", label="CV AUC")
    ax.fill_between(
        x,
        np.array(lc["cv_auc_mean"]) - np.array(lc["cv_auc_std"]),
        np.array(lc["cv_auc_mean"]) + np.array(lc["cv_auc_std"]),
        alpha=0.2,
    )
    ax.set_xlabel("Training Samples")
    ax.set_ylabel("ROC-AUC")
    ax.set_title("Preventive Model Learning Curve")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()


def run(csv_path: Path, output_dir: Path):
    df = pd.read_csv(csv_path)
    all_features = [c for c in df.columns if c not in [TARGET, SCORE_COL]]
    preventive_features = [c for c in all_features if c not in SYMPTOM_FEATURES]
    baseline_features = [c for c in BASELINE_FEATURES if c in df.columns]

    df = df[all_features + [TARGET]].dropna(subset=[TARGET])
    y = _target_binary(df[TARGET])

    configs = {
        "model_a_full": all_features,
        "model_b_preventive": preventive_features,
        "model_c_baseline": baseline_features,
    }

    summary = {}
    for name, feats in configs.items():
        X = df[feats].copy()
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
        )
        pipe = _build_pipeline(X_train, y_train)
        pipe.fit(X_train, y_train)

        train_proba = pipe.predict_proba(X_train)[:, 1]
        test_proba = pipe.predict_proba(X_test)[:, 1]
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        cv_scores = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc")

        summary[name] = {
            "n_features": len(feats),
            "features": feats,
            "train_auc": round(float(roc_auc_score(y_train, train_proba)), 4),
            "test_default_threshold": _metrics(y_test, test_proba, threshold=0.5),
            "test_recall_tuned": _best_recall_threshold(y_test, test_proba, max_precision_drop=0.04),
            "cv_roc_auc_mean": round(float(np.mean(cv_scores)), 4),
            "cv_roc_auc_std": round(float(np.std(cv_scores)), 4),
        }

        if name == "model_b_preventive":
            lc = _learning_curve_auc(pipe, X, y)
            summary[name]["learning_curve"] = lc
            _plot_learning_curve(
                lc, output_dir / "figures" / "preventive_learning_curve_auc.png"
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "tables").mkdir(parents=True, exist_ok=True)
    (output_dir / "figures").mkdir(parents=True, exist_ok=True)
    (output_dir / "preventive_model_comparison.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    _plot_bar(summary, "roc_auc", output_dir / "figures" / "preventive_auc_comparison.png")
    _plot_bar(summary, "recall", output_dir / "figures" / "preventive_recall_comparison.png")

    lines = [
        "\\begin{table}[h]",
        "\\centering",
        "\\caption{Leakage Audit: Full vs Preventive vs Baseline}",
        "\\label{tab:preventive_models}",
        "\\begin{tabular}{lcccccc}",
        "\\hline",
        "Model & Features & Train AUC & Test AUC & CV AUC & Recall@0.5 & Brier \\\\",
        "\\hline",
    ]
    for key in ["model_a_full", "model_b_preventive", "model_c_baseline"]:
        row = summary[key]
        td = row["test_default_threshold"]
        lines.append(
            f"{key.replace('_', ' ').title()} & {row['n_features']} & {row['train_auc']:.3f} & "
            f"{td['roc_auc']:.3f} & {row['cv_roc_auc_mean']:.3f} & {td['recall']:.3f} & {td['brier_score']:.3f} \\\\"
        )
    lines += ["\\hline", "\\end{tabular}", "\\end{table}"]
    (output_dir / "tables" / "preventive_model_comparison.tex").write_text(
        "\n".join(lines), encoding="utf-8"
    )

    print("Saved:", output_dir / "preventive_model_comparison.json")
    print("Saved:", output_dir / "tables" / "preventive_model_comparison.tex")
    print("Saved figures in:", output_dir / "figures")
    print("Main model (B) test AUC:", summary["model_b_preventive"]["test_default_threshold"]["roc_auc"])
    print("Main model (B) CV AUC:", summary["model_b_preventive"]["cv_roc_auc_mean"])


def main():
    parser = argparse.ArgumentParser(
        description="Build transparent Full/Preventive/Baseline models to mitigate leakage risk"
    )
    parser.add_argument("--csv", type=Path, required=True, help="Path to dataset.csv")
    parser.add_argument("--output-dir", type=Path, required=True, help="Results directory")
    args = parser.parse_args()
    run(args.csv, args.output_dir)


if __name__ == "__main__":
    main()
