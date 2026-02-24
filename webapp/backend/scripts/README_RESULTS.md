# Results Generation Scripts for IEEE Paper

This directory contains scripts to generate comprehensive results, visualizations, and tables for your IEEE paper submission.

## Quick Start

```bash
# Install dependencies
pip install -r requirements_results.txt

# Generate all results (recommended)
python generate_all_results.py --csv ../../oversampled_data.csv --output-dir ../../results

# Or run scripts individually:
python generate_results.py --csv ../../oversampled_data.csv --output-dir ../../results
python generate_shap_visualizations.py --model ../../model/risk_model.joblib --csv ../../oversampled_data.csv
python generate_fairness_analysis.py --output-dir ../../results
```

## Scripts Overview

### 1. `generate_results.py`
**Purpose**: Generate core performance metrics and visualizations

**Outputs**:
- `figures/confusion_matrix.png` - Confusion matrix heatmap
- `figures/roc_curve.png` - ROC curve with AUC score
- `figures/pr_curve.png` - Precision-Recall curve
- `tables/performance_metrics.tex` - LaTeX table for paper
- `metrics.json` - Complete metrics in JSON format

**Key Metrics Calculated**:
- Accuracy
- Weighted Precision, Recall, F1-Score
- ROC-AUC Score
- Per-class metrics
- Confusion Matrix

### 2. `generate_shap_visualizations.py`
**Purpose**: Generate SHAP explainability visualizations

**Outputs**:
- `figures/shap_feature_importance.png` - Global feature importance bar chart
- `figures/shap_summary.png` - SHAP summary plot showing feature impact distribution
- `figures/shap_waterfall.png` - Waterfall plot for single patient explanation
- `tables/feature_importance.tex` - LaTeX table of feature rankings

**Features**:
- Global feature importance using mean absolute SHAP values
- Local explanations via waterfall plots
- Feature impact distribution analysis

### 3. `generate_fairness_analysis.py`
**Purpose**: Generate fairness and bias analysis visualizations

**Outputs**:
- `figures/disparate_impact_analysis.png` - Disparate impact ratio visualization
- `figures/bias_comparison.png` - Bias comparison across protected groups
- `figures/mitigation_effectiveness.png` - Before/after mitigation comparison (optional)
- `tables/fairness_analysis.tex` - LaTeX table of fairness metrics
- `fairness_results.json` - Fairness analysis results

**Metrics**:
- Disparate Impact (DI) Ratio
- Bias Detection (DI < 0.8 threshold)
- Mitigation Strategy Recommendations

### 4. `generate_all_results.py`
**Purpose**: Master script that runs all analysis scripts

**Usage**:
```bash
python generate_all_results.py --csv ../../oversampled_data.csv --output-dir ../../results
```

## Output Structure

```
results/
├── figures/
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── pr_curve.png
│   ├── shap_feature_importance.png
│   ├── shap_summary.png
│   ├── shap_waterfall.png
│   ├── disparate_impact_analysis.png
│   ├── bias_comparison.png
│   └── mitigation_effectiveness.png (if provided)
├── tables/
│   ├── performance_metrics.tex
│   ├── feature_importance.tex
│   └── fairness_analysis.tex
├── metrics.json
├── fairness_results.json
└── RESULTS_SUMMARY.md
```

## Using Results in Your Paper

### Performance Metrics Section
1. Include the LaTeX table from `tables/performance_metrics.tex`
2. Reference `figures/confusion_matrix.png` to discuss:
   - False Negatives (critical in healthcare)
   - Model's ability to correctly identify high-risk patients
3. Include `figures/roc_curve.png` to show:
   - Model discrimination ability
   - AUC score comparison

### Explainability Section
1. Use `figures/shap_feature_importance.png` to show:
   - Which features contribute most to risk predictions
   - Clinical interpretability of the model
2. Include `figures/shap_waterfall.png` as a case study:
   - How the model arrived at a specific prediction
   - Feature-level contributions for a single patient

### Fairness Section
1. Include `figures/disparate_impact_analysis.png` to show:
   - Disparate Impact Ratio
   - Bias detection results
2. Use `tables/fairness_analysis.tex` to present:
   - Positive rates by protected group
   - Mitigation strategy recommendations

## Customization

### Custom Sample Input for Waterfall Plot
Create a JSON file with sample patient data:
```json
{
  "Age": 28,
  "Gestational Age": 20,
  "Trouble falling or staying sleep or sleeping too much": 3,
  "Poor appetite or overeating": 2,
  "Feeling tired or having little energy": 3,
  "Sufficient Money for Basic Needs": 1,
  "Thoughts that you would be better off dead, or of hurting yourself": 2
}
```

Then run:
```bash
python generate_shap_visualizations.py --sample-input sample_patient.json
```

### Custom Fairness Groups
Create a JSON file with your groups:
```json
[
  {"group": "low_income", "positive_rate": 0.25},
  {"group": "middle_income", "positive_rate": 0.30},
  {"group": "high_income", "positive_rate": 0.35}
]
```

Then run:
```bash
python generate_fairness_analysis.py --input groups.json
```

## Troubleshooting

### SHAP Import Error
If you get `ImportError: No module named 'shap'`:
```bash
pip install shap
```

### Matplotlib Backend Issues
If plots don't display, set backend:
```python
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
```

### Memory Issues with Large Datasets
Reduce sample size in SHAP calculations by modifying `sample_size` parameter in `generate_shap_visualizations.py`.

## Citation

When using these visualizations in your paper, cite:
- SHAP: Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. NIPS.
- Disparate Impact: Barocas, S., Hardt, M., & Narayanan, A. (2019). Fairness and Machine Learning.
