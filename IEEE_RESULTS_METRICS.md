# IEEE-Ready Results Section Metrics

## 1. Final Pooled (Leakage-Controlled) Model Metrics

### Publication Model (LightGBM Preventive Main)
*Source: results/publication_model_selection.json*

| Metric | Value |
|--------|-------|
| **Test ROC–AUC** | 0.5591 (95% CI: 0.5275–0.5897) |
| **5-fold CV ROC–AUC (mean ± std)** | 0.5519 ± 0.0111 |
| **Accuracy** | 0.5927 (59.27%) |
| **Precision (weighted)** | 0.6779 |
| **Recall** | 0.707 |
| **F1-score (weighted)** | 0.6922 |
| **Brier score** | 0.2534 |
| **Calibration status** | Uncalibrated (isotonic calibrated version available) |
| **Decision threshold** | 0.474 |

### Dataset Information
- **Dataset size after dedup**: 7,008 samples (from 14,008 original)
- **Class distribution**: Positive 64.8% (n=4,541), Negative 34.9% (n=2,467)

---

## 2. Confusion Matrix (Final Model - Main Test Set)

### Uncalibrated Model (from publication_model_selection.json)
| | Predicted Negative | Predicted Positive |
|---|---|---|
| **Actual Negative** | TN=189 | FP=305 |
| **Actual Positive** | FN=266 | TP=642 |

**Calculated Metrics:**
- **Sensitivity (Recall)**: 70.7%
- **Specificity**: 38.3%
- **Balanced Accuracy**: 54.5%

### Isotonic Calibrated Version
| | Predicted Negative | Predicted Positive |
|---|---|---|
| **Actual Negative** | TN=21 | FP=473 |
| **Actual Positive** | FN=9 | TP=899 |

- **Sensitivity**: 99.0%
- **Specificity**: 4.2%

---

## 3. Calibration Information

| Metric | Before Calibration | After Isotonic |
|--------|---------------------|----------------|
| **Brier Score** | 0.2534 | 0.2237 |
| **AUC** | 0.5591 | 0.5647 |
| **Threshold used** | 0.474 | N/A |

- **Threshold 0.5 used?** No, optimized threshold was 0.474

---

## 4. Source-Holdout Results (Tuned Version)

*Source: webapp/results/source_holdout_evaluation_tuned.json*

### By Source:
| Source | Train Size | Test Size | ROC–AUC | F1 (weighted) | Recall | Threshold |
|--------|------------|-----------|---------|---------------|--------|-----------|
| **main** | 11,287 | 6,903 | 0.6883 | 0.8042 | 0.5003 | 0.64 |
| **ppd** | 17,528 | 662 | 0.6714 | 0.7909 | 0.8772 | 0.53 |
| **uganda** | 7,565 | 10,625 | 0.5101 | 0.7682 | 0.0007 | 0.405 |

### Summary Statistics:
- **Mean AUC**: 0.6233 ± 0.0803
- **Mean Accuracy**: 0.6686 ± 0.0489
- **Mean F1**: 0.4603 ± 0.3287
- **Threshold fixed across domains?** No — domain-specific thresholds required

---

## 5. Random Forest (Sensitivity Model)

*Source: results/publication_model_selection.json - sensitivity_model*

| Metric | Value |
|--------|-------|
| **Test ROC–AUC** | 0.5597 |
| **Accuracy** | 0.6598 |
| **Precision** | 0.6626 |
| **Recall** | 0.967 |
| **F1 (weighted)** | 0.7864 |
| **Brier Score** | 0.2292 |
| **Threshold** | 0.4419 |

**Confusion Matrix:**
| | Predicted Negative | Predicted Positive |
|---|---|---|
| **Actual Negative** | 47 | 447 |
| **Actual Positive** | 30 | 878 |

**Recommendation:** Keep as sensitivity analysis or move to appendix (high recall but low specificity)

---

## 6. Figures Available

| Figure | File Path | Status |
|--------|-----------|--------|
| ROC curve (pooled) | results/figures/publication_main_roc_curve.png | ✅ Available |
| PR curve | results/figures/publication_main_pr_curve.png | ✅ Available |
| Calibration curve | results/figures/publication_calibration_curve.png | ✅ Available |
| Confusion matrix | results/figures/publication_main_confusion_matrix.png | ✅ Available |
| Source-holdout comparison | results/figures/publication_model_selection_auc.png | ✅ Available |
| Fairness plots | webapp/results/figures/bias_comparison.png | ✅ Available |
| Fairness plots | webapp/results/figures/disparate_impact_analysis.png | ✅ Available |

---

## 7. Fairness Metrics

*Source: webapp/results/fairness_results.json*

| Metric | Value |
|--------|-------|
| **Disparate Impact Ratio** | 0.714 |
| **Bias Detected** | Yes (DI < 0.8) |
| **Positive rate - Low income** | 25% |
| **Positive rate - Middle income** | 30% |
| **Positive rate - High income** | 35% |
| **Mitigation Strategy** | Reweighting |

---

## 8. Screening-Assisted Model (For Comparison)

*Source: results/publication_model_selection.json*

| Metric | Value |
|--------|-------|
| **Test ROC–AUC** | 0.8629 |
| **CV AUC (mean ± std)** | 0.8556 ± 0.0065 |
| **Accuracy** | 0.7846 |
| **Precision** | 0.8143 |
| **Recall** | 0.8645 |
| **F1 (weighted)** | 0.8387 |
| **Brier Score** | 0.1506 |

*Note: Includes concurrent symptom items and is not preventive-only.*

---

## 9. Important Notes for Results Section

1. **Overfitting Evidence**: Training AUC (~98%) vs CV AUC (~55%) indicates significant overfitting
2. **Domain Shift**: Source-holdout performance varies significantly (AUC 0.51-0.69)
3. **Preventive vs Screening**: Adding concurrent symptom data improves AUC from 0.56 to 0.86
4. **Label Noise**: Sensitivity analysis with high-confidence labels shows CV AUC of 0.59

---

*Generated from project results files*

