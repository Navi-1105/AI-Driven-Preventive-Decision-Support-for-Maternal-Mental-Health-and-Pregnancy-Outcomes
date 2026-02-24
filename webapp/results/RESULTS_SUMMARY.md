# Results Summary for IEEE Paper

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
