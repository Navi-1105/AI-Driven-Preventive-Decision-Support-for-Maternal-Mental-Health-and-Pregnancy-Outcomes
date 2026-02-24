import json
import matplotlib.pyplot as plt
import numpy as np
import os

# -------- Load fairness JSON --------
with open("fairness_results.json", "r") as f:
    fairness = json.load(f)

# Extract group rates (JSON structure is different - groups is an array)
groups_data = fairness["groups"]
groups = [g["group"] for g in groups_data]
rates = [g["positive_rate"] for g in groups_data]

di = fairness["disparate_impact"]
threshold = 0.8  # Standard fairness threshold

# -------- Plot --------
plt.figure(figsize=(8, 5))

bars = plt.bar(groups, rates)

# Add fairness threshold line
mean_rate = np.mean(rates)
plt.axhline(y=mean_rate * threshold, linestyle="--", color='red', label=f'Threshold ({threshold * 100}% of mean)')

# Annotate values on bars
for i, v in enumerate(rates):
    plt.text(i, v + 0.01, f"{v:.3f}", ha='center', fontsize=12)

plt.title(f"Fairness Analysis Across Income Groups\nDisparate Impact = {di:.3f}", fontsize=14)
plt.xlabel("Protected Group", fontsize=12)
plt.ylabel("Positive Prediction Rate", fontsize=12)

# Add bias text
if di < threshold:
    bias_text = "Bias Detected"
else:
    bias_text = "No Significant Bias"

plt.text(0.5, max(rates) + 0.08,
         f"DI = {di:.3f} | Threshold = {threshold}\n{bias_text}",
         ha='center', fontsize=11, 
         bbox=dict(boxstyle="round", facecolor="yellow" if di < threshold else "lightgreen", alpha=0.5))

plt.ylim(0, max(rates) + 0.2)
plt.legend()
plt.tight_layout()

# Save the plot
output_path = "fairness_plot.png"
plt.savefig(output_path, dpi=300)
print(f"Plot saved to {output_path}")
plt.show()

