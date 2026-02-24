"""
generate_fairness_analysis.py
Fairness and Bias Analysis Visualizations for IEEE Paper
Generates disparate impact analysis, bias detection charts, and mitigation reports
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

try:
    plt.style.use("seaborn-v0_8-paper")
except OSError:
    plt.style.use("seaborn-paper")  # Fallback for older matplotlib versions
sns.set_palette("husl")


def compute_disparate_impact(groups_data):
    """Compute disparate impact ratio."""
    rates = [g["positive_rate"] for g in groups_data if g["positive_rate"] > 0]
    if len(rates) < 2:
        return 1.0, False
    
    di = min(rates) / max(rates)
    bias_detected = di < 0.8
    return di, bias_detected


def plot_disparate_impact_analysis(groups_data, output_path: Path):
    """Generate disparate impact visualization."""
    groups = [g["group"] for g in groups_data]
    rates = [g["positive_rate"] for g in groups_data]
    di, bias_detected = compute_disparate_impact(groups_data)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Bar plot of positive rates
    colors = ["#e74c3c" if bias_detected else "#2ecc71" for _ in groups]
    bars = ax1.bar(groups, rates, color=colors, alpha=0.7, edgecolor="black", linewidth=1.5)
    ax1.set_xlabel("Protected Group", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Positive Rate", fontsize=12, fontweight="bold")
    ax1.set_title("Positive Rate by Protected Group", fontsize=14, fontweight="bold", pad=20)
    ax1.set_ylim([0, max(rates) * 1.2])
    ax1.grid(True, alpha=0.3, axis="y")

    # Add value labels
    for bar, rate in zip(bars, rates):
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{rate:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    # Add fairness threshold line
    if max(rates) > 0:
        threshold_rate = max(rates) * 0.8
        ax1.axhline(threshold_rate, color="orange", linestyle="--", linewidth=2, label="80% Threshold")
        ax1.legend(loc="upper right", fontsize=10)

    # Disparate Impact Ratio visualization
    di_color = "#e74c3c" if bias_detected else "#2ecc71"
    di_status = "Bias Detected" if bias_detected else "Fair"
    
    ax2.barh([0], [di], color=di_color, alpha=0.7, height=0.3, edgecolor="black", linewidth=1.5)
    ax2.axvline(0.8, color="orange", linestyle="--", linewidth=2, label="Fairness Threshold (0.8)")
    ax2.set_xlim([0, 1.1])
    ax2.set_xlabel("Disparate Impact Ratio", fontsize=12, fontweight="bold")
    ax2.set_title(f"Disparate Impact Analysis\nDI = {di:.3f} ({di_status})", fontsize=14, fontweight="bold", pad=20)
    ax2.set_yticks([])
    ax2.grid(True, alpha=0.3, axis="x")
    ax2.legend(loc="upper right", fontsize=10)

    # Add DI value label
    ax2.text(di, 0, f"{di:.3f}", ha="center", va="center", fontsize=14, fontweight="bold", color="white")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Disparate impact analysis saved to {output_path}")


def plot_bias_comparison(groups_data, output_path: Path):
    """Generate comparison chart showing bias across groups."""
    groups = [g["group"] for g in groups_data]
    rates = [g["positive_rate"] for g in groups_data]
    di, bias_detected = compute_disparate_impact(groups_data)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Create color map based on deviation from mean
    mean_rate = np.mean(rates)
    colors = []
    for rate in rates:
        deviation = abs(rate - mean_rate) / mean_rate if mean_rate > 0 else 0
        if deviation > 0.2:  # More than 20% deviation
            colors.append("#e74c3c")  # Red for high bias
        elif deviation > 0.1:  # 10-20% deviation
            colors.append("#f39c12")  # Orange for moderate bias
        else:
            colors.append("#2ecc71")  # Green for fair

    bars = ax.bar(groups, rates, color=colors, alpha=0.7, edgecolor="black", linewidth=1.5)

    # Add mean line
    ax.axhline(mean_rate, color="blue", linestyle="--", linewidth=2, label=f"Mean Rate: {mean_rate:.3f}")

    # Add fairness threshold (80% of max)
    if max(rates) > 0:
        threshold = max(rates) * 0.8
        ax.axhline(threshold, color="orange", linestyle="--", linewidth=2, label="Fairness Threshold (80%)")

    ax.set_xlabel("Protected Group", fontsize=12, fontweight="bold")
    ax.set_ylabel("Positive Rate", fontsize=12, fontweight="bold")
    ax.set_title(
        f"Bias Detection Across Protected Groups\nDisparate Impact: {di:.3f} ({'Bias Detected' if bias_detected else 'Fair'})",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")

    # Add value labels
    for bar, rate in zip(bars, rates):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{rate:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Bias comparison chart saved to {output_path}")


def plot_mitigation_effectiveness(before_di, after_di, output_path: Path):
    """Generate visualization showing mitigation effectiveness."""
    fig, ax = plt.subplots(figsize=(10, 6))

    categories = ["Before Mitigation", "After Mitigation"]
    di_values = [before_di, after_di]
    colors = ["#e74c3c" if di < 0.8 else "#2ecc71" for di in di_values]

    bars = ax.bar(categories, di_values, color=colors, alpha=0.7, edgecolor="black", linewidth=1.5, width=0.5)
    ax.axhline(0.8, color="orange", linestyle="--", linewidth=2, label="Fairness Threshold (0.8)")

    ax.set_ylabel("Disparate Impact Ratio", fontsize=12, fontweight="bold")
    ax.set_title("Mitigation Strategy Effectiveness", fontsize=14, fontweight="bold", pad=20)
    ax.set_ylim([0, 1.1])
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")

    # Add value labels
    for bar, di in zip(bars, di_values):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{di:.3f}",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )

    # Add improvement annotation
    improvement = after_di - before_di
    if improvement > 0:
        ax.annotate(
            f"Improvement: +{improvement:.3f}",
            xy=(1, after_di),
            xytext=(0.5, after_di + 0.1),
            arrowprops=dict(arrowstyle="->", color="green", lw=2),
            fontsize=11,
            fontweight="bold",
            color="green",
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Mitigation effectiveness plot saved to {output_path}")


def generate_fairness_table(groups_data, output_path: Path):
    """Generate LaTeX table for fairness metrics."""
    di, bias_detected = compute_disparate_impact(groups_data)

    table_rows = []
    table_rows.append("\\begin{table}[h]")
    table_rows.append("\\centering")
    table_rows.append("\\caption{Fairness Analysis: Disparate Impact by Protected Group}")
    table_rows.append("\\label{tab:fairness_analysis}")
    table_rows.append("\\begin{tabular}{lcc}")
    table_rows.append("\\hline")
    table_rows.append("Protected Group & Positive Rate & Status \\\\")
    table_rows.append("\\hline")

    max_rate = max(g["positive_rate"] for g in groups_data)
    threshold = max_rate * 0.8

    for group_data in groups_data:
        group = group_data["group"]
        rate = group_data["positive_rate"]
        status = "Fair" if rate >= threshold else "Bias"
        table_rows.append(f"{group} & {rate:.3f} & {status} \\\\")

    table_rows.append("\\hline")
    table_rows.append(f"\\textbf{{Disparate Impact}} & \\multicolumn{{2}}{{c}}{{{di:.3f}}} \\\\")
    table_rows.append(f"\\textbf{{Bias Detected}} & \\multicolumn{{2}}{{c}}{{{'Yes' if bias_detected else 'No'}}} \\\\")
    table_rows.append(f"\\textbf{{Mitigation Strategy}} & \\multicolumn{{2}}{{c}}{{{'Reweighting' if bias_detected else 'Monitor Only'}}} \\\\")
    table_rows.append("\\hline")
    table_rows.append("\\end{tabular}")
    table_rows.append("\\end{table}")

    output_path.write_text("\n".join(table_rows), encoding="utf-8")
    print(f"✓ Fairness table saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate fairness analysis visualizations for IEEE paper")
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="JSON file with groups data (if not provided, uses example data)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("../../results"),
        help="Output directory for visualizations",
    )
    parser.add_argument(
        "--before-di",
        type=float,
        default=None,
        help="Disparate impact before mitigation (for comparison plot)",
    )
    parser.add_argument(
        "--after-di",
        type=float,
        default=None,
        help="Disparate impact after mitigation (for comparison plot)",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "figures").mkdir(exist_ok=True)
    (args.output_dir / "tables").mkdir(exist_ok=True)

    print("=" * 60)
    print("Generating Fairness and Bias Analysis Visualizations")
    print("=" * 60)

    # Load or create example groups data
    input_file = args.input or Path(__file__).parent / "sample_fairness_groups.json"
    if input_file.exists():
        print("\n1. Loading groups data from file...")
        groups_data = json.loads(input_file.read_text())
    else:
        print("\n1. Using example groups data...")
        # Example data: income-based groups
        groups_data = [
            {"group": "low_income", "positive_rate": 0.25},
            {"group": "middle_income", "positive_rate": 0.30},
            {"group": "high_income", "positive_rate": 0.35},
        ]

    print(f"   Groups: {[g['group'] for g in groups_data]}")
    di, bias_detected = compute_disparate_impact(groups_data)
    print(f"   Disparate Impact: {di:.3f}")
    print(f"   Bias Detected: {bias_detected}")

    # Generate visualizations
    print("\n2. Generating fairness visualizations...")

    # Disparate impact analysis
    plot_disparate_impact_analysis(
        groups_data, args.output_dir / "figures" / "disparate_impact_analysis.png"
    )

    # Bias comparison
    plot_bias_comparison(groups_data, args.output_dir / "figures" / "bias_comparison.png")

    # Fairness table
    generate_fairness_table(groups_data, args.output_dir / "tables" / "fairness_analysis.tex")

    # Mitigation effectiveness (if before/after provided)
    if args.before_di is not None and args.after_di is not None:
        print("\n3. Generating mitigation effectiveness plot...")
        plot_mitigation_effectiveness(
            args.before_di, args.after_di, args.output_dir / "figures" / "mitigation_effectiveness.png"
        )

    # Save results JSON
    results = {
        "groups": groups_data,
        "disparate_impact": di,
        "bias_detected": bias_detected,
        "mitigation_strategy": "reweighting" if bias_detected else "monitor_only",
    }
    results_path = args.output_dir / "fairness_results.json"
    results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"✓ Fairness results saved to {results_path}")

    print("\n" + "=" * 60)
    print("Fairness Analysis Complete")
    print(f"Results saved to: {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
