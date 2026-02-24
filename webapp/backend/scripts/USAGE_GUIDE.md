# Usage Guide: Results Generation for IEEE Paper

## Quick Start

### Option 1: Generate Everything at Once (Recommended)
```bash
cd webapp/backend/scripts
python generate_all_results.py --csv ../../oversampled_data.csv --output-dir ../../results
```

### Option 2: Run Individual Scripts
```bash
# 1. Performance Metrics
python generate_results.py --csv ../../oversampled_data.csv --output-dir ../../results

# 2. SHAP Visualizations (requires trained model)
python generate_shap_visualizations.py --model ../../model/risk_model.joblib --csv ../../oversampled_data.csv --output-dir ../../results

# 3. Fairness Analysis
python generate_fairness_analysis.py --output-dir ../../results
```

## What Each Script Generates

### 1. Performance Metrics (`generate_results.py`)

**Outputs:**
- ✅ Confusion Matrix (PNG, 300 DPI)
- ✅ ROC Curve (PNG, 300 DPI)  
- ✅ Precision-Recall Curve (PNG, 300 DPI)
- ✅ LaTeX Table of Metrics
- ✅ JSON Metrics File

**Key Metrics:**
- Overall Accuracy
- Weighted Precision, Recall, F1-Score
- ROC-AUC Score
- Per-class metrics
- Confusion Matrix values

**How to Use in Paper:**
- **Table**: Copy LaTeX table from `tables/performance_metrics.tex` into your Results section
- **Figure**: Reference `figures/confusion_matrix.png` to discuss False Negatives
- **ROC Curve**: Include `figures/roc_curve.png` to show model discrimination ability

### 2. SHAP Explainability (`generate_shap_visualizations.py`)

**Outputs:**
- ✅ Global Feature Importance Bar Chart
- ✅ SHAP Summary Plot (feature impact distribution)
- ✅ Waterfall Plot (local explanation for single patient)
- ✅ LaTeX Table of Feature Rankings

**Key Visualizations:**
- **Feature Importance**: Shows which features contribute most to predictions
- **Summary Plot**: Shows distribution of SHAP values across dataset
- **Waterfall Plot**: Shows how model arrived at specific prediction for one patient

**How to Use in Paper:**
- **Section III-C**: Include feature importance plot to show explainability
- **Case Study**: Use waterfall plot to demonstrate local explanation
- **Table**: Include feature importance rankings table

### 3. Fairness Analysis (`generate_fairness_analysis.py`)

**Outputs:**
- ✅ Disparate Impact Analysis Chart
- ✅ Bias Comparison Across Groups
- ✅ Mitigation Effectiveness (if before/after data provided)
- ✅ LaTeX Table of Fairness Metrics

**Key Metrics:**
- Disparate Impact (DI) Ratio
- Bias Detection (DI < 0.8 threshold)
- Mitigation Strategy Recommendations

**How to Use in Paper:**
- **Section on Fairness**: Include disparate impact visualization
- **Table**: Include fairness metrics table
- **Discussion**: Reference bias detection and mitigation strategies

## Customization Examples

### Custom Patient Sample for Waterfall Plot

Create `custom_patient.json`:
```json
{
  "Age": 32,
  "Gestational Age": 28,
  "Trouble falling or staying sleep or sleeping too much": 2,
  "Poor appetite or overeating": 1,
  "Feeling tired or having little energy": 2,
  "Sufficient Money for Basic Needs": 2,
  "Thoughts that you would be better off dead, or of hurting yourself": 0
}
```

Run:
```bash
python generate_shap_visualizations.py \
    --model ../../model/risk_model.joblib \
    --csv ../../oversampled_data.csv \
    --sample-input custom_patient.json \
    --output-dir ../../results
```

### Custom Fairness Groups

Create `custom_groups.json`:
```json
[
  {"group": "rural", "positive_rate": 0.28},
  {"group": "urban", "positive_rate": 0.32}
]
```

Run:
```bash
python generate_fairness_analysis.py \
    --input custom_groups.json \
    --output-dir ../../results
```

### Before/After Mitigation Comparison

```bash
python generate_fairness_analysis.py \
    --before-di 0.65 \
    --after-di 0.85 \
    --output-dir ../../results
```

## Output File Structure

```
results/
├── figures/
│   ├── confusion_matrix.png          # Performance visualization
│   ├── roc_curve.png                 # ROC curve
│   ├── pr_curve.png                  # Precision-Recall curve
│   ├── shap_feature_importance.png   # Global feature importance
│   ├── shap_summary.png              # SHAP summary plot
│   ├── shap_waterfall.png            # Local explanation
│   ├── disparate_impact_analysis.png # Fairness visualization
│   └── bias_comparison.png           # Bias comparison
├── tables/
│   ├── performance_metrics.tex       # LaTeX table for metrics
│   ├── feature_importance.tex        # LaTeX table for features
│   └── fairness_analysis.tex         # LaTeX table for fairness
├── metrics.json                      # Complete metrics (JSON)
├── fairness_results.json             # Fairness results (JSON)
└── RESULTS_SUMMARY.md                # Summary document
```

## Integration with Paper

### Results Section Structure

1. **Performance Metrics**
   - Include Table from `tables/performance_metrics.tex`
   - Reference Figure `figures/confusion_matrix.png`
   - Discuss ROC-AUC from `figures/roc_curve.png`

2. **Explainability Results**
   - Include `figures/shap_feature_importance.png`
   - Use `figures/shap_waterfall.png` for case study
   - Reference Table from `tables/feature_importance.tex`

3. **Fairness Analysis**
   - Include `figures/disparate_impact_analysis.png`
   - Use Table from `tables/fairness_analysis.tex`
   - Discuss mitigation strategies

## Troubleshooting

### Import Errors
```bash
pip install -r requirements_results.txt
```

### SHAP Not Working
```bash
pip install shap>=0.42.0
```

### Matplotlib Backend Issues
The scripts automatically use non-interactive backend for saving figures.

### Memory Issues
Reduce sample size in `generate_shap_visualizations.py`:
```python
sample_size = min(50, len(X_train))  # Reduce from 100 to 50
```

## Citation Guidelines

When using these visualizations, cite:

1. **SHAP**: 
   - Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. NIPS.

2. **Disparate Impact**:
   - Barocas, S., Hardt, M., & Narayanan, A. (2019). Fairness and Machine Learning.

3. **Scikit-learn**:
   - Pedregosa et al. (2011). Scikit-learn: Machine Learning in Python. JMLR.

## Next Steps

1. Run `generate_all_results.py` to create all visualizations
2. Review outputs in `results/` directory
3. Copy LaTeX tables into your paper
4. Include figures with appropriate captions
5. Reference metrics in your Results section
