/**
 * ShapExplanation.jsx
 * SHAP Feature-Level Impact Visualization Component
 * Shows feature contributions to risk score with interactive tooltips
 * Paper Section: III-C. SHAP Explainability
 */

export default function ShapExplanation({ contributions = [], riskPercent = null, crisisMode = false }) {
  if (crisisMode) {
    return (
      <div className="shap-explanation crisis">
        <div className="crisis-override">
          <p className="crisis-text">
            ⚠️ Crisis Protocol Active: XAI explanations hidden due to self-harm indicator detection.
          </p>
          <p className="muted">Immediate clinical intervention required.</p>
        </div>
      </div>
    );
  }

  if (!contributions || contributions.length === 0) {
    return (
      <div className="shap-explanation empty">
        <p className="muted">No SHAP contributions available. Run risk assessment to see feature explanations.</p>
      </div>
    );
  }

  // Sort by absolute contribution value (descending)
  const ranked = [...contributions].sort(
    (a, b) => Math.abs(b.contribution_percent) - Math.abs(a.contribution_percent)
  );

  // Calculate total positive and negative contributions
  const positiveContrib = ranked
    .filter((c) => c.contribution_percent > 0)
    .reduce((sum, c) => sum + c.contribution_percent, 0);
  const negativeContrib = ranked
    .filter((c) => c.contribution_percent < 0)
    .reduce((sum, c) => sum + Math.abs(c.contribution_percent), 0);

  return (
    <div className="shap-explanation">
      <div className="shap-header">
        <h3>Feature Impact Analysis</h3>
        {riskPercent !== null && (
          <div className="shap-summary">
            <span className="shap-summary-item positive">
              +{positiveContrib.toFixed(1)}% positive factors
            </span>
            <span className="shap-summary-item negative">
              -{negativeContrib.toFixed(1)}% protective factors
            </span>
          </div>
        )}
      </div>

      <div className="shap-contributions">
        {ranked.slice(0, 8).map((item, idx) => {
          const isPositive = item.contribution_percent > 0;
          const absValue = Math.abs(item.contribution_percent);
          const featureName = item.feature.replace(/_/g, " ").replace(/\b\w/g, (l) => l.toUpperCase());

          return (
            <div key={`${item.feature}-${idx}`} className="shap-row">
              <div className="shap-row-header">
                <span className="shap-feature-name">{featureName}</span>
                <span className={`shap-value ${isPositive ? "positive" : "negative"}`}>
                  {isPositive ? "+" : ""}
                  {item.contribution_percent.toFixed(1)}%
                </span>
              </div>
              <div className="shap-bar-container">
                <div
                  className={`shap-bar ${isPositive ? "positive" : "negative"}`}
                  style={{
                    width: `${Math.min(100, absValue)}%`,
                    marginLeft: isPositive ? "0" : "auto",
                    marginRight: isPositive ? "auto" : "0"
                  }}
                >
                  <div className="shap-bar-fill" />
                </div>
              </div>
            </div>
          );
        })}
      </div>

      <div className="shap-footer">
        <p className="muted">
          Values show how each feature contributes to the risk score. Positive values increase risk, negative values
          decrease risk.
        </p>
      </div>
    </div>
  );
}
