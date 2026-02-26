"""
generate_shap_visualizations.py
SHAP Explainability Visualizations for IEEE Paper
Generates feature importance plots and waterfall plots for local explanations
"""

import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb

DEFAULT_FEATURES = [
    "Age",
    "Gestational Age",
    "Trouble falling or staying sleep or sleeping too much",
    "Poor appetite or overeating",
    "Feeling tired or having little energy",
    "Sufficient Money for Basic Needs",
    "Thoughts that you would be better off dead, or of hurting yourself",
]

try:
    plt.style.use("seaborn-v0_8-paper")
except OSError:
    plt.style.use("seaborn-paper")  # Fallback for older matplotlib versions


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


def _get_estimator_and_matrix(model_payload, X_df: pd.DataFrame):
    pipeline = model_payload["pipeline"]
    estimator = None
    for step_name in ("xgb", "rf"):
        if step_name in pipeline.named_steps:
            estimator = pipeline.named_steps[step_name]
            break
    if estimator is None:
        estimator = pipeline.steps[-1][1]

    if len(pipeline.steps) > 1:
        preprocess = pipeline[:-1]
        X_model = preprocess.transform(X_df)
    else:
        X_model = X_df.values
    return estimator, X_model


def _compute_shap_matrix(model_payload, X_df: pd.DataFrame) -> np.ndarray:
    features = model_payload["features"]
    estimator, X_model = _get_estimator_and_matrix(model_payload, X_df)

    # Prefer native XGBoost SHAP contributions when available.
    if estimator.__class__.__name__.lower().startswith("xgb"):
        booster = estimator.get_booster()
        dmat = xgb.DMatrix(X_model, feature_names=features)
        contribs = booster.predict(dmat, pred_contribs=True)
        return np.asarray(contribs)[:, : len(features)]

    try:
        explainer = shap.TreeExplainer(estimator)
        shap_values = explainer.shap_values(X_model)
    except Exception:
        explainer = shap.Explainer(estimator)
        shap_values = explainer(X_model).values

    if isinstance(shap_values, list):
        class_idx = 1 if len(shap_values) > 1 else 0
        shap_values_class = np.array(shap_values[class_idx])
    else:
        shap_values_class = np.array(shap_values)

    if shap_values_class.ndim == 3:
        shap_values_class = shap_values_class[:, :, 0]
    elif shap_values_class.ndim > 2:
        shap_values_class = shap_values_class.reshape(shap_values_class.shape[0], -1)
    return shap_values_class


def load_model(model_path: Path):
    """Load trained model."""
    payload = joblib.load(model_path)
    return payload


def plot_global_feature_importance(model_payload, X_train, output_path: Path):
    """Generate global feature importance plot using SHAP."""
    features = model_payload["features"]
    
    # Calculate SHAP values for a sample of training data
    sample_size = min(100, len(X_train))
    X_sample = X_train.sample(n=sample_size, random_state=42) if isinstance(X_train, pd.DataFrame) else X_train[:sample_size]
    shap_values_class = _compute_shap_matrix(model_payload, X_sample)

    # Calculate mean absolute SHAP values for global importance
    mean_shap = np.abs(shap_values_class).mean(axis=0)

    # Create feature importance plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Sort features by importance
    indices = np.argsort(mean_shap)[::-1]
    # Convert features to numpy array for proper indexing
    features_array = np.array(features)
    sorted_features = features_array[indices].tolist()
    sorted_importance = mean_shap[indices]

    # Create horizontal bar plot
    y_pos = np.arange(len(sorted_features))
    bars = ax.barh(y_pos, sorted_importance, color="steelblue", alpha=0.7)
    
    # Customize plot
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f.replace("_", " ").title() for f in sorted_features], fontsize=10)
    ax.set_xlabel("Mean |SHAP Value|", fontsize=12, fontweight="bold")
    ax.set_title("Global Feature Importance (SHAP)", fontsize=14, fontweight="bold", pad=20)
    ax.grid(True, alpha=0.3, axis="x")

    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, sorted_importance)):
        ax.text(val + 0.01, i, f"{val:.3f}", va="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Global feature importance plot saved to {output_path}")


def plot_shap_summary(model_payload, X_train, output_path: Path):
    """Generate SHAP summary plot."""
    features = model_payload["features"]

    sample_size = min(100, len(X_train))
    X_sample = X_train.sample(n=sample_size, random_state=42) if isinstance(X_train, pd.DataFrame) else X_train[:sample_size]
    shap_values_class = _compute_shap_matrix(model_payload, X_sample)

    # Create summary plot
    fig = plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values_class, X_sample, feature_names=features, show=False)
    plt.title("SHAP Summary Plot: Feature Impact Distribution", fontsize=14, fontweight="bold", pad=20)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ SHAP summary plot saved to {output_path}")


def plot_waterfall_explanation(model_payload, sample_input, output_path: Path):
    """Generate waterfall plot for a single patient explanation."""
    features = model_payload["features"]
    classes = model_payload.get("label_classes", [])

    # Prepare input
    X_sample = pd.DataFrame([sample_input], columns=features)
    shap_vals = _compute_shap_matrix(model_payload, X_sample)[0]

    # Handle 3D array (1 sample x features x classes)
    if shap_vals.ndim == 2:
        shap_vals = shap_vals[:, 0]
    elif shap_vals.ndim > 1:
        shap_vals = shap_vals.flatten()

    # Estimate base value and prediction from the full pipeline.
    pipeline = model_payload["pipeline"]
    base_value = 0.0
    prediction = pipeline.predict_proba(X_sample)[0]
    if len(prediction) > 1:
        if "Depressed" in classes:
            class_idx = classes.index("Depressed")
        else:
            class_idx = 0
        pred_value = prediction[class_idx]
    else:
        pred_value = prediction[0]

    # Create waterfall data
    feature_names = [f.replace("_", " ").title() for f in features]
    contributions = shap_vals.tolist()

    # Sort by absolute contribution
    sorted_indices = np.argsort(np.abs(contributions))[::-1]
    sorted_features = [feature_names[i] for i in sorted_indices]
    sorted_contributions = [contributions[i] for i in sorted_indices]

    # Create waterfall plot
    fig, ax = plt.subplots(figsize=(10, 8))

    # Calculate cumulative values
    cumulative = base_value
    positions = []
    values = []
    colors = []

    for i, (feat, contrib) in enumerate(zip(sorted_features, sorted_contributions)):
        start = cumulative
        end = cumulative + contrib
        positions.append(i)
        values.append((start, end))
        colors.append("red" if contrib > 0 else "blue")
        cumulative = end

    # Plot bars
    for i, (pos, (start, end), color) in enumerate(zip(positions, values, colors)):
        ax.barh(pos, end - start, left=start, color=color, alpha=0.7, height=0.6)
        # Add value label
        mid = (start + end) / 2
        ax.text(mid, pos, f"{end-start:.3f}", va="center", ha="center", fontsize=9, fontweight="bold")

    # Add base value and prediction lines
    ax.axvline(base_value, color="gray", linestyle="--", linewidth=2, label=f"Base Value: {base_value:.3f}")
    ax.axvline(pred_value, color="green", linestyle="--", linewidth=2, label=f"Prediction: {pred_value:.3f}")

    # Customize plot
    ax.set_yticks(positions)
    ax.set_yticklabels(sorted_features, fontsize=10)
    ax.set_xlabel("SHAP Value", fontsize=12, fontweight="bold")
    ax.set_title("Waterfall Plot: Local Explanation for Single Patient", fontsize=14, fontweight="bold", pad=20)
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Waterfall plot saved to {output_path}")


def generate_feature_importance_table(model_payload, X_train, output_path: Path):
    """Generate LaTeX table for feature importance."""
    features = model_payload["features"]

    sample_size = min(100, len(X_train))
    X_sample = X_train.sample(n=sample_size, random_state=42) if isinstance(X_train, pd.DataFrame) else X_train[:sample_size]
    shap_values_class = _compute_shap_matrix(model_payload, X_sample)
    
    # Calculate mean absolute SHAP values
    mean_shap = np.abs(shap_values_class).mean(axis=0)
    
    # Ensure we have 1D array for mean SHAP values
    if mean_shap.ndim > 1:
        mean_shap = mean_shap.flatten()
    
    indices = np.argsort(mean_shap)[::-1]
    
    # Convert features to numpy array for proper indexing
    features_array = np.array(features)

    table_rows = []
    table_rows.append("\\begin{table}[h]")
    table_rows.append("\\centering")
    table_rows.append("\\caption{Feature Importance Ranking (SHAP Values)}")
    table_rows.append("\\label{tab:feature_importance}")
    table_rows.append("\\begin{tabular}{lc}")
    table_rows.append("\\hline")
    table_rows.append("Feature & Mean |SHAP Value| \\\\")
    table_rows.append("\\hline")

    for idx in indices:
        feature_name = str(features_array[idx]).replace("_", " ").title()
        importance = float(mean_shap[idx])  # Convert to scalar
        table_rows.append(f"{feature_name} & {importance:.4f}\\\\")

    table_rows.append("\\hline")
    table_rows.append("\\end{tabular}")
    table_rows.append("\\end{table}")

    output_path.write_text("\n".join(table_rows), encoding="utf-8")
    print(f"✓ Feature importance table saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate SHAP visualizations for IEEE paper")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("../../model/risk_model.joblib"),
        help="Path to trained model",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("../../dataset.csv"),
        help="Path to training CSV for SHAP calculations",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("../../results"),
        help="Output directory for visualizations",
    )
    parser.add_argument(
        "--sample-input",
        type=Path,
        default=None,
        help="JSON file with sample input for waterfall plot (optional)",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "figures").mkdir(exist_ok=True)
    (args.output_dir / "tables").mkdir(exist_ok=True)

    print("=" * 60)
    print("Generating SHAP Explainability Visualizations")
    print("=" * 60)

    # Load model
    print("\n1. Loading model...")
    model_payload = load_model(args.model)
    print(f"   Model accuracy: {model_payload['accuracy']:.4f}")

    # Load training data for SHAP calculations
    print("\n2. Loading training data...")
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder

    df = pd.read_csv(args.csv)
    df = df[DEFAULT_FEATURES + [model_payload["target"]]]
    df[DEFAULT_FEATURES] = _coerce_feature_frame(df[DEFAULT_FEATURES])
    df = df.dropna(subset=DEFAULT_FEATURES + [model_payload["target"]])
    X = df[DEFAULT_FEATURES]
    y_raw = df[model_payload["target"]].astype(str).str.strip()
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)
    X_train, X_test, _, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Generate visualizations
    print("\n3. Generating SHAP visualizations...")

    # Global feature importance
    plot_global_feature_importance(
        model_payload, X_train, args.output_dir / "figures" / "shap_feature_importance.png"
    )

    # SHAP summary plot
    plot_shap_summary(model_payload, X_train, args.output_dir / "figures" / "shap_summary.png")

    # Feature importance table
    generate_feature_importance_table(
        model_payload, X_train, args.output_dir / "tables" / "feature_importance.tex"
    )

    # Waterfall plot (if sample input provided)
    sample_input_file = args.sample_input or Path(__file__).parent / "sample_high_risk_patient.json"
    if sample_input_file.exists():
        print("\n4. Generating waterfall plot for sample input...")
        sample_data = json.loads(sample_input_file.read_text())
        sample_input = [sample_data.get(f, 0) for f in DEFAULT_FEATURES]
        plot_waterfall_explanation(
            model_payload, sample_input, args.output_dir / "figures" / "shap_waterfall.png"
        )
    else:
        # Use a default high-risk sample
        print("\n4. Generating waterfall plot for default high-risk sample...")
        default_sample = {
            "Age": 28,
            "Gestational Age": 20,
            "Trouble falling or staying sleep or sleeping too much": 3,
            "Poor appetite or overeating": 2,
            "Feeling tired or having little energy": 3,
            "Sufficient Money for Basic Needs": 1,
            "Thoughts that you would be better off dead, or of hurting yourself": 2,
        }
        sample_input = [default_sample.get(f, 0) for f in DEFAULT_FEATURES]
        plot_waterfall_explanation(
            model_payload, sample_input, args.output_dir / "figures" / "shap_waterfall.png"
        )

    print("\n" + "=" * 60)
    print("SHAP Visualizations Complete")
    print(f"Results saved to: {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
