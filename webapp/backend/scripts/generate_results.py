"""
generate_results.py
Comprehensive leakage-safe results generation aligned with production XGBoost training.
"""

import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

try:
    from imblearn.over_sampling import SMOTE
except Exception:
    SMOTE = None

import warnings

warnings.filterwarnings("ignore")

# Set style for IEEE-compliant figures
try:
    plt.style.use("seaborn-v0_8-paper")
except OSError:
    plt.style.use("seaborn-paper")
sns.set_palette("husl")

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
ROOT_DIR = Path(__file__).resolve().parents[3]
BACKEND_DIR = Path(__file__).resolve().parents[1]


def bootstrap_ci_auc(y_true, y_proba, n_bootstraps=1000, confidence_level=0.95):
    n_samples = len(y_true)
    rng = np.random.RandomState(RANDOM_STATE)
    auc_scores = []

    for _ in range(n_bootstraps):
        indices = rng.randint(0, n_samples, n_samples)
        if len(np.unique(y_true[indices])) < 2:
            continue
        auc_scores.append(roc_auc_score(y_true[indices], y_proba[indices]))

    auc_scores = np.array(auc_scores)
    alpha = 1 - confidence_level
    return {
        "auc": round(float(np.mean(auc_scores)), 4),
        "ci_lower": round(float(np.percentile(auc_scores, 100 * alpha / 2)), 4),
        "ci_upper": round(float(np.percentile(auc_scores, 100 * (1 - alpha / 2))), 4),
        "se": round(float(np.std(auc_scores)), 4),
        "n_bootstraps": n_bootstraps,
    }


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


def load_and_prepare_data(csv_path: Path):
    df = pd.read_csv(csv_path)
    df = df[DEFAULT_FEATURES + [TARGET]]
    df[DEFAULT_FEATURES] = _coerce_feature_frame(df[DEFAULT_FEATURES])
    df = df.dropna(subset=DEFAULT_FEATURES + [TARGET])
    X = df[DEFAULT_FEATURES]
    y_raw = df[TARGET].astype(str).str.strip()
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)
    return X, y, label_encoder, df


def train_model(X, y, balance_strategy="smote"):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    X_train_bal, y_train_bal, applied_balance = _balance_train(X_train, y_train, balance_strategy)
    pipeline = _build_pipeline()
    pipeline.fit(X_train_bal, y_train_bal)
    return pipeline, X_train, X_test, y_train, y_test, y_train_bal, applied_balance


def calculate_cv_metrics(X, y, folds=5):
    if folds < 2:
        return {}
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_validate(
        _build_pipeline(),
        X,
        y,
        cv=cv,
        scoring={"roc_auc": "roc_auc", "f1_weighted": "f1_weighted"},
    )
    return {
        "folds": folds,
        "roc_auc_mean": round(float(np.mean(cv_scores["test_roc_auc"])), 4),
        "roc_auc_std": round(float(np.std(cv_scores["test_roc_auc"])), 4),
        "f1_weighted_mean": round(float(np.mean(cv_scores["test_f1_weighted"])), 4),
        "f1_weighted_std": round(float(np.std(cv_scores["test_f1_weighted"])), 4),
    }


def calculate_metrics(y_test, y_pred, y_proba, label_encoder):
    classes = label_encoder.classes_
    y_pos = y_proba[:, 1] if y_proba.shape[1] > 1 else y_proba[:, 0]

    metrics = {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "classification_report": classification_report(
            y_test, y_pred, target_names=list(classes), output_dict=True
        ),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }

    pr, rc, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="weighted")
    metrics["precision_weighted"] = round(float(pr), 4)
    metrics["recall_weighted"] = round(float(rc), 4)
    metrics["f1_weighted"] = round(float(f1), 4)

    pr_per_class, rc_per_class, f1_per_class, _ = precision_recall_fscore_support(
        y_test, y_pred, average=None, labels=range(len(classes))
    )
    metrics["per_class"] = {
        class_name: {
            "precision": round(float(pr_per_class[i]), 4),
            "recall": round(float(rc_per_class[i]), 4),
            "f1_score": round(float(f1_per_class[i]), 4),
        }
        for i, class_name in enumerate(classes)
    }

    if len(classes) == 2:
        metrics["roc_auc"] = round(float(roc_auc_score(y_test, y_pos)), 4)
        metrics["pr_auc"] = round(float(average_precision_score(y_test, y_pos)), 4)
        metrics["brier_score"] = round(float(brier_score_loss(y_test, y_pos)), 4)
        auc_ci = bootstrap_ci_auc(y_test, y_pos, n_bootstraps=1000)
        metrics["roc_auc_ci"] = {
            "lower": auc_ci["ci_lower"],
            "upper": auc_ci["ci_upper"],
            "se": auc_ci["se"],
            "n_bootstraps": auc_ci["n_bootstraps"],
        }
        fpr, tpr, _ = roc_curve(y_test, y_pos)
        metrics["roc_curve"] = {"fpr": fpr.tolist(), "tpr": tpr.tolist()}
        precision, recall, _ = precision_recall_curve(y_test, y_pos)
        metrics["pr_curve"] = {"precision": precision.tolist(), "recall": recall.tolist()}

    return metrics


def plot_confusion_matrix(y_test, y_pred, label_encoder, output_path: Path):
    cm = confusion_matrix(y_test, y_pred)
    classes = label_encoder.classes_
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
        ax=ax,
        cbar_kws={"label": "Count"},
    )
    ax.set_xlabel("Predicted Label", fontsize=12, fontweight="bold")
    ax.set_ylabel("True Label", fontsize=12, fontweight="bold")
    ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold", pad=20)
    ax.text(
        0.5,
        -0.15,
        f"Overall Accuracy: {accuracy_score(y_test, y_pred):.3f}",
        transform=ax.transAxes,
        ha="center",
        fontsize=11,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_roc_curve(y_test, y_proba, output_path: Path):
    y_pos = y_proba[:, 1] if y_proba.shape[1] > 1 else y_proba[:, 0]
    fpr, tpr, _ = roc_curve(y_test, y_pos)
    roc_auc = roc_auc_score(y_test, y_pos)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random Classifier")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=12, fontweight="bold")
    ax.set_ylabel("True Positive Rate", fontsize=12, fontweight="bold")
    ax.set_title("Receiver Operating Characteristic (ROC) Curve", fontsize=14, fontweight="bold", pad=20)
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_precision_recall_curve(y_test, y_proba, output_path: Path):
    y_pos = y_proba[:, 1] if y_proba.shape[1] > 1 else y_proba[:, 0]
    precision, recall, _ = precision_recall_curve(y_test, y_pos)
    pr_auc = average_precision_score(y_test, y_pos)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall, precision, color="darkgreen", lw=2, label=f"PR curve (AUC = {pr_auc:.3f})")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Recall", fontsize=12, fontweight="bold")
    ax.set_ylabel("Precision", fontsize=12, fontweight="bold")
    ax.set_title("Precision-Recall Curve", fontsize=14, fontweight="bold", pad=20)
    ax.legend(loc="lower left", fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def generate_metrics_table(metrics, output_path: Path):
    classes = list(metrics["classification_report"].keys())
    classes = [c for c in classes if c not in ["accuracy", "macro avg", "weighted avg"]]

    rows = [
        "\\begin{table}[h]",
        "\\centering",
        "\\caption{Model Performance Metrics (Leakage-Safe XGBoost)}",
        "\\label{tab:performance_metrics}",
        "\\begin{tabular}{lcccc}",
        "\\hline",
        "Class & Precision & Recall & F1-Score & Support \\\\",
        "\\hline",
    ]

    for class_name in classes:
        report = metrics["classification_report"].get(class_name, {})
        rows.append(
            f"{class_name} & {report.get('precision', 0):.3f} & {report.get('recall', 0):.3f} & "
            f"{report.get('f1-score', 0):.3f} & {int(report.get('support', 0))} \\\\"
        )

    rows.extend(
        [
            "\\hline",
            f"\\textbf{{Weighted Avg}} & {metrics['precision_weighted']:.3f} & {metrics['recall_weighted']:.3f} & {metrics['f1_weighted']:.3f} & - \\\\",
            "\\hline",
            f"\\textbf{{Accuracy}} & \\multicolumn{{4}}{{c}}{{{metrics['accuracy']:.3f}}} \\\\",
            f"\\textbf{{ROC-AUC}} & \\multicolumn{{4}}{{c}}{{{metrics.get('roc_auc', 0.0):.3f}}} \\\\",
            f"\\textbf{{PR-AUC}} & \\multicolumn{{4}}{{c}}{{{metrics.get('pr_auc', 0.0):.3f}}} \\\\",
            f"\\textbf{{Brier Score}} & \\multicolumn{{4}}{{c}}{{{metrics.get('brier_score', 0.0):.4f}}} \\\\",
        ]
    )
    if "cross_validation" in metrics:
        cv = metrics["cross_validation"]
        rows.append(
            f"\\textbf{{CV ROC-AUC}} & \\multicolumn{{4}}{{c}}{{{cv['roc_auc_mean']:.4f} $\\pm$ {cv['roc_auc_std']:.4f}}} \\\\"
        )
    rows.extend(["\\hline", "\\end{tabular}", "\\end{table}"])

    output_path.write_text("\n".join(rows), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Generate leakage-safe XGBoost results for paper")
    parser.add_argument(
        "--csv",
        type=Path,
        default=ROOT_DIR / "dataset.csv",
        help="Path to raw training CSV",
    )
    parser.add_argument(
        "--balance-strategy",
        type=str,
        default="smote",
        choices=["smote", "random", "none"],
        help="Class balancing strategy on training split only",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Stratified CV folds for stability reporting",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT_DIR / "results",
        help="Output directory for results and visualizations",
    )
    parser.add_argument(
        "--model-out",
        type=Path,
        default=BACKEND_DIR / "model/risk_model.joblib",
        help="Output model file",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "figures").mkdir(exist_ok=True)
    (args.output_dir / "tables").mkdir(exist_ok=True)

    X, y, label_encoder, df = load_and_prepare_data(args.csv)
    pipeline, X_train, X_test, y_train, y_test, y_train_bal, applied_balance = train_model(
        X, y, args.balance_strategy
    )
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)
    metrics = calculate_metrics(y_test, y_pred, y_proba, label_encoder)
    metrics["cross_validation"] = calculate_cv_metrics(X, y, folds=args.cv_folds)
    metrics["reproducibility"] = {
        "random_state": RANDOM_STATE,
        "test_size": 0.2,
        "train_size": 0.8,
        "feature_order": DEFAULT_FEATURES,
        "balance_strategy_requested": args.balance_strategy,
        "balance_strategy_applied": applied_balance,
        "class_balance_before_train": {str(k): int(v) for k, v in pd.Series(y_train).value_counts().to_dict().items()},
        "class_balance_after_train_balancing": {str(k): int(v) for k, v in pd.Series(y_train_bal).value_counts().to_dict().items()},
        "dataset_rows_after_dropna": int(df.shape[0]),
    }

    metrics_json_path = args.output_dir / "metrics.json"
    metrics_json_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    plot_confusion_matrix(
        y_test, y_pred, label_encoder, args.output_dir / "figures" / "confusion_matrix.png"
    )
    if len(label_encoder.classes_) == 2:
        plot_roc_curve(y_test, y_proba, args.output_dir / "figures" / "roc_curve.png")
        plot_precision_recall_curve(y_test, y_proba, args.output_dir / "figures" / "pr_curve.png")
    generate_metrics_table(metrics, args.output_dir / "tables" / "performance_metrics.tex")

    model_payload = {
        "pipeline": pipeline,
        "features": DEFAULT_FEATURES,
        "label_classes": list(label_encoder.classes_),
        "target": TARGET,
        "accuracy": metrics["accuracy"],
        "metrics": metrics,
        "reproducibility": metrics["reproducibility"],
    }
    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_payload, args.model_out)

    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"ROC-AUC: {metrics.get('roc_auc', 0.0):.4f}")
    print(f"PR-AUC: {metrics.get('pr_auc', 0.0):.4f}")
    print(f"Brier Score: {metrics.get('brier_score', 0.0):.4f}")
    if metrics.get("cross_validation"):
        cv = metrics["cross_validation"]
        print(f"CV ROC-AUC: {cv['roc_auc_mean']:.4f} +/- {cv['roc_auc_std']:.4f}")
    print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
