"""
generate_results.py
Comprehensive Results Generation Script for IEEE Paper
Generates performance metrics, SHAP visualizations, and fairness analysis
"""

import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

# Set style for IEEE-compliant figures
try:
    plt.style.use("seaborn-v0_8-paper")
except OSError:
    plt.style.use("seaborn-paper")  # Fallback for older matplotlib versions
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


def load_and_prepare_data(csv_path: Path):
    """Load and prepare data for training."""
    df = pd.read_csv(csv_path)
    df = df[DEFAULT_FEATURES + [TARGET]].dropna()

    X = df[DEFAULT_FEATURES]
    y_raw = df[TARGET].astype(str).str.strip()

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)

    return X, y, label_encoder, df


def train_model(X, y, random_state=42):
    """Train Random Forest model."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("rf", RandomForestClassifier(n_estimators=300, random_state=random_state, n_jobs=-1)),
        ]
    )

    pipeline.fit(X_train, y_train)
    return pipeline, X_train, X_test, y_train, y_test


def calculate_metrics(y_test, y_pred, y_proba, label_encoder):
    """Calculate comprehensive performance metrics."""
    classes = label_encoder.classes_
    
    metrics = {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "classification_report": classification_report(
            y_test, y_pred, target_names=list(classes), output_dict=True
        ),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }

    # Weighted metrics
    pr, rc, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="weighted")
    metrics["precision_weighted"] = round(float(pr), 4)
    metrics["recall_weighted"] = round(float(rc), 4)
    metrics["f1_weighted"] = round(float(f1), 4)

    # Per-class metrics
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

    # ROC-AUC for binary classification
    if len(classes) == 2:
        metrics["roc_auc"] = round(float(roc_auc_score(y_test, y_proba[:, 1])), 4)
        metrics["pr_auc"] = round(float(average_precision_score(y_test, y_proba[:, 1])), 4)
        # Calculate ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1])
        metrics["roc_curve"] = {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
        }
        # Calculate Precision-Recall curve
        precision, recall, _ = precision_recall_curve(y_test, y_proba[:, 1])
        metrics["pr_curve"] = {
            "precision": precision.tolist(),
            "recall": recall.tolist(),
        }

    return metrics


def plot_confusion_matrix(y_test, y_pred, label_encoder, output_path: Path):
    """Generate confusion matrix visualization."""
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

    # Add accuracy text
    accuracy = accuracy_score(y_test, y_pred)
    ax.text(
        0.5,
        -0.15,
        f"Overall Accuracy: {accuracy:.3f}",
        transform=ax.transAxes,
        ha="center",
        fontsize=11,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Confusion matrix saved to {output_path}")


def plot_roc_curve(y_test, y_proba, output_path: Path):
    """Generate ROC curve visualization."""
    fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1])
    roc_auc = roc_auc_score(y_test, y_proba[:, 1])

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
    print(f"✓ ROC curve saved to {output_path}")


def plot_precision_recall_curve(y_test, y_proba, output_path: Path):
    """Generate Precision-Recall curve visualization."""
    precision, recall, _ = precision_recall_curve(y_test, y_proba[:, 1])
    pr_auc = average_precision_score(y_test, y_proba[:, 1])

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
    print(f"✓ Precision-Recall curve saved to {output_path}")


def generate_metrics_table(metrics, output_path: Path):
    """Generate LaTeX table for IEEE paper."""
    classes = list(metrics["classification_report"].keys())
    classes = [c for c in classes if c not in ["accuracy", "macro avg", "weighted avg"]]

    table_rows = []
    table_rows.append("\\begin{table}[h]")
    table_rows.append("\\centering")
    table_rows.append("\\caption{Model Performance Metrics}")
    table_rows.append("\\label{tab:performance_metrics}")
    table_rows.append("\\begin{tabular}{lcccc}")
    table_rows.append("\\hline")
    table_rows.append("Class & Precision & Recall & F1-Score & Support \\\\")
    table_rows.append("\\hline")

    for class_name in classes:
        if isinstance(class_name, str) and class_name.replace("_", " ").replace(".", "").isdigit():
            continue
        report = metrics["classification_report"].get(class_name, {})
        precision = report.get("precision", 0)
        recall = report.get("recall", 0)
        f1 = report.get("f1-score", 0)
        support = report.get("support", 0)
        table_rows.append(
            f"{class_name} & {precision:.3f} & {recall:.3f} & {f1:.3f} & {int(support)} \\\\"
        )

    table_rows.append("\\hline")
    table_rows.append(
        f"\\textbf{{Weighted Avg}} & {metrics['precision_weighted']:.3f} & "
        f"{metrics['recall_weighted']:.3f} & {metrics['f1_weighted']:.3f} & - \\\\"
    )
    table_rows.append("\\hline")
    table_rows.append(f"\\textbf{{Accuracy}} & \\multicolumn{{4}}{{c}}{{{metrics['accuracy']:.3f}}} \\\\")
    if "roc_auc" in metrics:
        table_rows.append(f"\\textbf{{ROC-AUC}} & \\multicolumn{{4}}{{c}}{{{metrics['roc_auc']:.3f}}} \\\\")
    table_rows.append("\\hline")
    table_rows.append("\\end{tabular}")
    table_rows.append("\\end{table}")

    output_path.write_text("\n".join(table_rows), encoding="utf-8")
    print(f"✓ LaTeX table saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate comprehensive results for IEEE paper")
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("../../oversampled_data.csv"),
        help="Path to training CSV",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("../../results"),
        help="Output directory for results and visualizations",
    )
    parser.add_argument(
        "--model-out",
        type=Path,
        default=Path("../../model/risk_model.joblib"),
        help="Output model file",
    )
    args = parser.parse_args()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "figures").mkdir(exist_ok=True)
    (args.output_dir / "tables").mkdir(exist_ok=True)

    print("=" * 60)
    print("Generating Comprehensive Results for IEEE Paper")
    print("=" * 60)

    # Load data
    print("\n1. Loading and preparing data...")
    X, y, label_encoder, df = load_and_prepare_data(args.csv)
    print(f"   Dataset shape: {df.shape}")
    print(f"   Classes: {label_encoder.classes_}")

    # Train model
    print("\n2. Training Random Forest model...")
    pipeline, X_train, X_test, y_train, y_test = train_model(X, y)

    # Predictions
    print("\n3. Generating predictions...")
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)

    # Calculate metrics
    print("\n4. Calculating performance metrics...")
    metrics = calculate_metrics(y_test, y_pred, y_proba, label_encoder)

    # Save metrics JSON
    metrics_json_path = args.output_dir / "metrics.json"
    metrics_json_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"✓ Metrics saved to {metrics_json_path}")

    # Generate visualizations
    print("\n5. Generating visualizations...")

    # Confusion Matrix
    plot_confusion_matrix(
        y_test, y_pred, label_encoder, args.output_dir / "figures" / "confusion_matrix.png"
    )

    # ROC Curve (for binary classification)
    if len(label_encoder.classes_) == 2:
        plot_roc_curve(y_test, y_proba, args.output_dir / "figures" / "roc_curve.png")
        plot_precision_recall_curve(y_test, y_proba, args.output_dir / "figures" / "pr_curve.png")

    # Metrics Table
    generate_metrics_table(metrics, args.output_dir / "tables" / "performance_metrics.tex")

    # Save model
    print("\n6. Saving model...")
    model_payload = {
        "pipeline": pipeline,
        "features": DEFAULT_FEATURES,
        "label_classes": list(label_encoder.classes_),
        "target": TARGET,
        "accuracy": metrics["accuracy"],
        "metrics": metrics,
    }
    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_payload, args.model_out)
    print(f"✓ Model saved to {args.model_out}")

    # Print summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision (Weighted): {metrics['precision_weighted']:.4f}")
    print(f"Recall (Weighted): {metrics['recall_weighted']:.4f}")
    print(f"F1-Score (Weighted): {metrics['f1_weighted']:.4f}")
    if "roc_auc" in metrics:
        print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"\nResults saved to: {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
