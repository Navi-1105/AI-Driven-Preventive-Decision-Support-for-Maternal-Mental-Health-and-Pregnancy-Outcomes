"""
generate_all_results.py
Master script to generate all results and visualizations for IEEE paper
Runs all analysis scripts and creates comprehensive results package
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_script(script_path: Path, args: list):
    """Run a Python script and handle errors."""
    try:
        cmd = [sys.executable, str(script_path)] + args
        print(f"\n{'='*60}")
        print(f"Running: {' '.join(cmd)}")
        print(f"{'='*60}")
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("Warnings:", result.stderr)
        return True
    except subprocess.CalledProcessError as e:
        print(f"ERROR running {script_path.name}:")
        print(e.stderr)
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Generate all results and visualizations for IEEE paper"
    )
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
        help="Output directory for all results",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("../../model"),
        help="Model directory",
    )
    args = parser.parse_args()

    # Get script directory
    script_dir = Path(__file__).parent

    print("=" * 60)
    print("COMPREHENSIVE RESULTS GENERATION FOR IEEE PAPER")
    print("=" * 60)
    print(f"Output directory: {args.output_dir}")
    print(f"CSV file: {args.csv}")
    print(f"Model directory: {args.model_dir}")

    # Step 1: Generate performance metrics and visualizations
    print("\n" + "=" * 60)
    print("STEP 1: Generating Performance Metrics")
    print("=" * 60)
    success = run_script(
        script_dir / "generate_results.py",
        [
            "--csv",
            str(args.csv),
            "--output-dir",
            str(args.output_dir),
            "--model-out",
            str(args.model_dir / "risk_model.joblib"),
        ],
    )
    if not success:
        print("WARNING: Performance metrics generation had errors")

    # Step 2: Generate SHAP visualizations
    print("\n" + "=" * 60)
    print("STEP 2: Generating SHAP Explainability Visualizations")
    print("=" * 60)
    success = run_script(
        script_dir / "generate_shap_visualizations.py",
        [
            "--model",
            str(args.model_dir / "risk_model.joblib"),
            "--csv",
            str(args.csv),
            "--output-dir",
            str(args.output_dir),
        ],
    )
    if not success:
        print("WARNING: SHAP visualization generation had errors")

    # Step 3: Generate fairness analysis
    print("\n" + "=" * 60)
    print("STEP 3: Generating Fairness Analysis")
    print("=" * 60)
    success = run_script(
        script_dir / "generate_fairness_analysis.py",
        [
            "--output-dir",
            str(args.output_dir),
        ],
    )
    if not success:
        print("WARNING: Fairness analysis generation had errors")

    # Create summary report
    print("\n" + "=" * 60)
    print("STEP 4: Creating Summary Report")
    print("=" * 60)
    
    summary_path = args.output_dir / "RESULTS_SUMMARY.md"
    summary_content = f"""# Results Summary for IEEE Paper

## Generated Files

### Performance Metrics
- `figures/confusion_matrix.png` - Confusion matrix visualization
- `figures/roc_curve.png` - ROC curve (binary classification)
- `figures/pr_curve.png` - Precision-Recall curve
- `tables/performance_metrics.tex` - LaTeX table of performance metrics
- `metrics.json` - Complete metrics in JSON format

### SHAP Explainability
- `figures/shap_feature_importance.png` - Global feature importance
- `figures/shap_summary.png` - SHAP summary plot
- `figures/shap_waterfall.png` - Waterfall plot for local explanation
- `tables/feature_importance.tex` - LaTeX table of feature importance

### Fairness Analysis
- `figures/disparate_impact_analysis.png` - Disparate impact visualization
- `figures/bias_comparison.png` - Bias comparison across groups
- `tables/fairness_analysis.tex` - LaTeX table of fairness metrics
- `fairness_results.json` - Fairness analysis results

## Usage in Paper

1. **Performance Metrics**: Include Table from `tables/performance_metrics.tex` in Results section
2. **Confusion Matrix**: Reference Figure `figures/confusion_matrix.png` to discuss False Negatives
3. **ROC Curve**: Include `figures/roc_curve.png` to show model discrimination ability
4. **SHAP Feature Importance**: Use `figures/shap_feature_importance.png` to show which features drive predictions
5. **Waterfall Plot**: Include `figures/shap_waterfall.png` for case study explanation
6. **Fairness Analysis**: Include `figures/disparate_impact_analysis.png` and Table from `tables/fairness_analysis.tex`

## Key Metrics to Report

- Overall Accuracy
- Weighted Precision, Recall, F1-Score
- ROC-AUC Score (for binary classification)
- Disparate Impact Ratio
- Feature Importance Rankings (SHAP values)

All visualizations are saved at 300 DPI for publication quality.
"""
    
    summary_path.write_text(summary_content, encoding="utf-8")
    print(f"✓ Summary report saved to {summary_path}")

    print("\n" + "=" * 60)
    print("ALL RESULTS GENERATION COMPLETE")
    print("=" * 60)
    print(f"\nResults are available in: {args.output_dir}")
    print(f"Review the summary: {summary_path}")


if __name__ == "__main__":
    main()
