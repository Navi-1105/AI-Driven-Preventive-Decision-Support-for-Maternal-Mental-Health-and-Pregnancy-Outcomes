# Results Generation Summary

## Overview

This document summarizes the comprehensive results generation system created for your IEEE paper submission. The system generates all necessary performance metrics, visualizations, and tables required for a strong Results section.

## Created Scripts

### 1. `generate_results.py` - Core Performance Metrics
**Purpose**: Generate comprehensive performance metrics and visualizations

**Generates**:
- Confusion Matrix (PNG, 300 DPI)
- ROC Curve with AUC score
- Precision-Recall Curve
- LaTeX table of all metrics
- JSON file with complete metrics

**Key Metrics Calculated**:
- ✅ Accuracy
- ✅ Weighted Precision, Recall, F1-Score
- ✅ ROC-AUC Score (for binary classification)
- ✅ Per-class metrics
- ✅ Confusion Matrix (critical for False Negative analysis)

**Usage**:
```bash
python generate_results.py --csv ../../oversampled_data.csv --output-dir ../../results
```

### 2. `generate_shap_visualizations.py` - Explainability Visualizations
**Purpose**: Generate SHAP-based explainability visualizations

**Generates**:
- Global Feature Importance Bar Chart
- SHAP Summary Plot (feature impact distribution)
- Waterfall Plot (local explanation for single patient)
- LaTeX table of feature importance rankings

**Key Features**:
- Shows which features contribute most to risk predictions
- Demonstrates local explanations for individual patients
- Provides feature-level SHAP contributions

**Usage**:
```bash
python generate_shap_visualizations.py \
    --model ../../model/risk_model.joblib \
    --csv ../../oversampled_data.csv \
    --output-dir ../../results
```

### 3. `generate_fairness_analysis.py` - Fairness and Bias Analysis
**Purpose**: Generate fairness analysis visualizations

**Generates**:
- Disparate Impact Analysis Chart
- Bias Comparison Across Protected Groups
- Mitigation Effectiveness Plot (optional)
- LaTeX table of fairness metrics

**Key Metrics**:
- ✅ Disparate Impact (DI) Ratio
- ✅ Bias Detection (DI < 0.8 threshold)
- ✅ Mitigation Strategy Recommendations

**Usage**:
```bash
python generate_fairness_analysis.py --output-dir ../../results
```

### 4. `generate_all_results.py` - Master Script
**Purpose**: Run all analysis scripts in sequence

**Usage**:
```bash
python generate_all_results.py \
    --csv ../../oversampled_data.csv \
    --output-dir ../../results
```

## Output Structure

```
results/
├── figures/
│   ├── confusion_matrix.png          # Performance: Confusion Matrix
│   ├── roc_curve.png                 # Performance: ROC Curve
│   ├── pr_curve.png                  # Performance: Precision-Recall
│   ├── shap_feature_importance.png   # XAI: Global Feature Importance
│   ├── shap_summary.png              # XAI: SHAP Summary Plot
│   ├── shap_waterfall.png            # XAI: Local Explanation
│   ├── disparate_impact_analysis.png # Fairness: DI Analysis
│   └── bias_comparison.png           # Fairness: Bias Comparison
├── tables/
│   ├── performance_metrics.tex       # LaTeX: Performance Table
│   ├── feature_importance.tex        # LaTeX: Feature Rankings
│   └── fairness_analysis.tex         # LaTeX: Fairness Table
├── metrics.json                      # Complete metrics (JSON)
├── fairness_results.json             # Fairness results (JSON)
└── RESULTS_SUMMARY.md                 # Auto-generated summary
```

## How to Use in Your IEEE Paper

### Results Section Structure

#### 1. Performance Metrics Subsection

**Include**:
- Table from `tables/performance_metrics.tex`
- Figure: `figures/confusion_matrix.png` (discuss False Negatives)
- Figure: `figures/roc_curve.png` (show AUC score)

**Example Text**:
> "Table I presents the comprehensive performance metrics of our Random Forest classifier. The model achieved an overall accuracy of X.XX% with weighted precision, recall, and F1-scores of X.XX, X.XX, and X.XX respectively. The ROC-AUC score of X.XX demonstrates strong discrimination ability between depressed and non-depressed states. Figure X shows the confusion matrix, highlighting the model's performance across classes. Notably, the False Negative rate of X.XX% is critical in healthcare applications, as it represents cases where high-risk patients were incorrectly classified as low-risk."

#### 2. Explainability Subsection

**Include**:
- Figure: `figures/shap_feature_importance.png`
- Figure: `figures/shap_waterfall.png` (case study)
- Table: `tables/feature_importance.tex`

**Example Text**:
> "Figure X presents the global feature importance derived from SHAP values, showing that 'Thoughts of self-harm' contributes X.XX% to the risk prediction, followed by 'Sleep disturbance' at X.XX%. Figure Y demonstrates a local explanation for a high-risk patient case using a waterfall plot, illustrating how each feature contributed to the final risk score of X.XX%."

#### 3. Fairness Analysis Subsection

**Include**:
- Figure: `figures/disparate_impact_analysis.png`
- Table: `tables/fairness_analysis.tex`

**Example Text**:
> "Table X presents the fairness analysis across income-based groups. The disparate impact ratio of X.XX indicates [fairness/bias status]. The positive rates for low-income, middle-income, and high-income groups were X.XX, X.XX, and X.XX respectively. [If bias detected:] Bias was detected (DI < 0.8), and reweighting mitigation strategies were recommended."

## Key Metrics to Report

### Performance Metrics
- **Accuracy**: Overall classification accuracy
- **Precision (Weighted)**: Accounts for class imbalance
- **Recall (Weighted)**: Critical for healthcare (minimize False Negatives)
- **F1-Score (Weighted)**: Balanced metric
- **ROC-AUC**: Model discrimination ability
- **Confusion Matrix**: Per-class performance breakdown

### Explainability Metrics
- **Feature Importance**: Mean absolute SHAP values
- **Contribution Percentages**: How much each feature contributes
- **Local Explanations**: Patient-specific feature contributions

### Fairness Metrics
- **Disparate Impact Ratio**: min(positive_rates) / max(positive_rates)
- **Bias Detection**: DI < 0.8 indicates bias
- **Mitigation Strategy**: Reweighting vs. Monitor Only

## Installation

```bash
cd webapp/backend/scripts
pip install -r requirements_results.txt
```

## Quick Start

```bash
# Generate all results
cd webapp/backend/scripts
python generate_all_results.py --csv ../../oversampled_data.csv --output-dir ../../results

# Or use the shell script
./run_results_generation.sh
```

## Customization

### Custom Patient Sample
Edit `sample_high_risk_patient.json` or create your own:
```json
{
  "Age": 28,
  "Gestational Age": 20,
  "Trouble falling or staying sleep or sleeping too much": 3,
  ...
}
```

### Custom Fairness Groups
Edit `sample_fairness_groups.json` or create your own:
```json
[
  {"group": "low_income", "positive_rate": 0.25},
  {"group": "high_income", "positive_rate": 0.35}
]
```

## Troubleshooting

### SHAP Import Error
```bash
pip install shap>=0.42.0
```

### Matplotlib Style Error
The scripts automatically fall back to `seaborn-paper` if `seaborn-v0_8-paper` is not available.

### Memory Issues
Reduce sample size in SHAP calculations by editing `sample_size` parameter.

## Citation

When using these visualizations, cite:
- **SHAP**: Lundberg & Lee (2017) - A unified approach to interpreting model predictions
- **Disparate Impact**: Barocas, Hardt, & Narayanan (2019) - Fairness and Machine Learning
- **Scikit-learn**: Pedregosa et al. (2011) - Scikit-learn: Machine Learning in Python

## Next Steps

1. ✅ Run `generate_all_results.py` to create all visualizations
2. ✅ Review outputs in `results/` directory
3. ✅ Copy LaTeX tables into your paper
4. ✅ Include figures with appropriate captions
5. ✅ Reference metrics in your Results section
6. ✅ Ensure all visualizations meet IEEE figure requirements (300 DPI, clear labels)

## Support Files

- `README_RESULTS.md` - Detailed documentation
- `USAGE_GUIDE.md` - Step-by-step usage instructions
- `requirements_results.txt` - Python dependencies
- `sample_high_risk_patient.json` - Example patient data
- `sample_fairness_groups.json` - Example fairness groups
