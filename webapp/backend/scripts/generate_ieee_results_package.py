import argparse
import json
from pathlib import Path

import joblib
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator
from sklearn.calibration import calibration_curve
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

import train_model_combined as t

matplotlib.use("Agg")


def _compute_predictions(root_dir: Path, model_path: Path):
    payload = joblib.load(model_path)
    pipeline = payload["pipeline"]
    threshold = float(payload.get("threshold", 0.5))

    full_df = t._assemble_dataset(
        root_dir / "dataset.csv",
        root_dir / "Data for Postpartum Depression Prediction in Bangladesh/PPD_dataset_v2.csv",
        root_dir / "Large Scale Anonymized EPDS Data for Prenatal Women in Selected hospitals in Uganda/records.csv",
        {"main", "ppd", "uganda"},
    )
    full_df = full_df.drop_duplicates(subset=t.COMMON_FEATURES + [t.TARGET]).reset_index(drop=True)
    X = full_df[t.COMMON_FEATURES].copy()
    y = full_df[t.TARGET].astype(int)
    groups = t._row_signature(full_df, t.COMMON_FEATURES + [t.TARGET])
    _, _, X_test, _, _, y_test = t._split_leakage_safe(
        X, y, groups, test_size=t.TEST_SIZE, val_size=t.VALIDATION_SIZE
    )

    proba = pipeline.predict_proba(X_test)[:, 1]
    pred = (proba >= threshold).astype(int)
    return y_test.to_numpy(), proba, pred, threshold


def _build_metrics(y_true, y_prob, y_pred, threshold):
    pr_w, rc_w, f1_w, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, y_prob)), 4),
        "pr_auc": round(float(average_precision_score(y_true, y_prob)), 4),
        "brier_score": round(float(brier_score_loss(y_true, y_prob)), 4),
        "precision_weighted": round(float(pr_w), 4),
        "recall_weighted": round(float(rc_w), 4),
        "f1_weighted": round(float(f1_w), 4),
        "classification_report": classification_report(y_true, y_pred, output_dict=True),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "threshold": round(float(threshold), 4),
        "evaluation_protocol": "leakage_safe_group_split_deduplicated",
    }


def _write_outputs(base_dir: Path, metrics: dict, y_true, y_prob, y_pred):
    fig_dir = base_dir / "figures"
    table_dir = base_dir / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    (base_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    rows = [
        "\\begin{table}[h]",
        "\\centering",
        "\\caption{Leakage-Safe Combined Model Performance (IEEE format)}",
        "\\label{tab:performance_metrics}",
        "\\begin{tabular}{lc}",
        "\\hline",
        "Metric & Value \\\\",
        "\\hline",
        f"Accuracy & {metrics['accuracy']:.4f} \\\\",
        f"ROC-AUC & {metrics['roc_auc']:.4f} \\\\",
        f"PR-AUC & {metrics['pr_auc']:.4f} \\\\",
        f"Brier Score & {metrics['brier_score']:.4f} \\\\",
        f"Precision (Weighted) & {metrics['precision_weighted']:.4f} \\\\",
        f"Recall (Weighted) & {metrics['recall_weighted']:.4f} \\\\",
        f"F1 (Weighted) & {metrics['f1_weighted']:.4f} \\\\",
        f"Decision Threshold & {metrics['threshold']:.3f} \\\\",
        "\\hline",
        "\\end{tabular}",
        "\\end{table}",
    ]
    (table_dir / "performance_metrics.tex").write_text("\n".join(rows), encoding="utf-8")

    cm = confusion_matrix(y_true, y_pred)
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    cal_true, cal_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy="quantile")
    plt.style.use("seaborn-v0_8-whitegrid")
    sns.set(style="whitegrid")
    plt.rcParams.update(
        {
            "font.size": 16,
            "axes.titlesize": 18,
            "axes.labelsize": 16,
            "legend.fontsize": 14,
            "figure.figsize": (8, 6),
            "lines.linewidth": 2.8,
        }
    )

    # ROC
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"AUC = {metrics['roc_auc']:.3f}")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1.2)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Receiver Operating Characteristic (ROC)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.xaxis.set_major_locator(MaxNLocator(6))
    ax.yaxis.set_major_locator(MaxNLocator(6))
    ax.legend(loc="lower right", frameon=True)
    ax.grid(True, alpha=0.35)
    fig.tight_layout()
    fig.savefig(fig_dir / "roc_curve.png", dpi=400)
    plt.close(fig)

    # PR
    fig, ax = plt.subplots()
    ax.plot(rec, prec, label=f"PR-AUC = {metrics['pr_auc']:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.xaxis.set_major_locator(MaxNLocator(6))
    ax.yaxis.set_major_locator(MaxNLocator(6))
    ax.legend(loc="lower left", frameon=True)
    ax.grid(True, alpha=0.35)
    fig.tight_layout()
    fig.savefig(fig_dir / "pr_curve.png", dpi=400)
    plt.close(fig)

    # Calibration
    fig, ax = plt.subplots()
    ax.plot(cal_pred, cal_true, marker="o", linewidth=2.5, label="Model")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1.2, label="Perfect calibration")
    ax.set_xlabel("Predicted Probability")
    ax.set_ylabel("Observed Frequency")
    ax.set_title("Calibration Curve")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.xaxis.set_major_locator(MaxNLocator(6))
    ax.yaxis.set_major_locator(MaxNLocator(6))
    ax.legend(loc="upper left", frameon=True)
    ax.grid(True, alpha=0.35)
    fig.tight_layout()
    fig.savefig(fig_dir / "calibration_curve.png", dpi=400)
    plt.close(fig)

    # Confusion matrix
    fig, ax = plt.subplots()
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        ax=ax,
        annot_kws={"size": 16, "weight": "bold"},
    )
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_xticklabels(["Not", "Depressed"], rotation=0)
    ax.set_yticklabels(["Not", "Depressed"], rotation=0)
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    fig.savefig(fig_dir / "confusion_matrix.png", dpi=400)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Generate labeled IEEE-style results package.")
    parser.add_argument(
        "--root-dir",
        type=Path,
        default=Path(__file__).resolve().parents[3],
        help="Project root directory.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "model" / "risk_model_combined_leakage_safe.joblib",
        help="Model payload path.",
    )
    parser.add_argument(
        "--output-dirs",
        type=str,
        default="results,webapp/results",
        help="Comma-separated output dirs relative to root.",
    )
    args = parser.parse_args()

    root_dir = args.root_dir.resolve()
    model_path = args.model_path if args.model_path.is_absolute() else (root_dir / args.model_path)
    outputs = [root_dir / p.strip() for p in args.output_dirs.split(",") if p.strip()]

    y_true, y_prob, y_pred, threshold = _compute_predictions(root_dir, model_path)
    metrics = _build_metrics(y_true, y_prob, y_pred, threshold)

    for out in outputs:
        _write_outputs(out, metrics, y_true, y_prob, y_pred)
        print(f"Updated: {out}")


if __name__ == "__main__":
    main()
