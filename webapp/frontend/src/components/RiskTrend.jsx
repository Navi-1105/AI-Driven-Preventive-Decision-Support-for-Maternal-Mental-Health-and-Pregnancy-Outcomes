export default function RiskTrend({ timelinePoints = [], currentRisk = null, gestationalWeeks = null }) {
  const safeRisk = typeof currentRisk === "number" ? Math.max(0, Math.min(100, currentRisk)) : null;
  const status = getRiskStatus(safeRisk);
  const history = [...timelinePoints]
    .sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp))
    .slice(0, 3);

  return (
    <div className="risk-status-panel">
      <h3 className="risk-trend-title">Risk Status</h3>

      <div className={`risk-status-card ${status.tone}`}>
        <span className="risk-status-label">{status.label}</span>
        <strong>{safeRisk === null ? "Not calculated" : `${safeRisk.toFixed(1)}%`}</strong>
        <p>{status.message}</p>
      </div>

      <div className="risk-band-scale" aria-label="Risk bands">
        <div className="risk-band low">
          <span>Low</span>
          <strong>0-40%</strong>
        </div>
        <div className="risk-band moderate">
          <span>Moderate</span>
          <strong>41-70%</strong>
        </div>
        <div className="risk-band high">
          <span>High</span>
          <strong>71-100%</strong>
        </div>
        {safeRisk !== null ? (
          <span className="risk-band-marker" style={{ left: `${safeRisk}%` }}>
            Current
          </span>
        ) : null}
      </div>

      <div className="risk-status-details">
        <div>
          <span>Gestational week</span>
          <strong>{gestationalWeeks ? `Week ${gestationalWeeks}` : "Not set"}</strong>
        </div>
        <div>
          <span>Stored readings</span>
          <strong>{timelinePoints.length}</strong>
        </div>
      </div>

      {history.length ? (
        <div className="risk-history-list">
          <span className="risk-history-title">Recent readings</span>
          {history.map((point, index) => (
            <div className="risk-history-row" key={`${point.timestamp}-${index}`}>
              <span>Week {point.gestational_weeks}</span>
              <strong>{Number(point.risk_percent).toFixed(1)}%</strong>
            </div>
          ))}
        </div>
      ) : (
        <p className="muted">Run risk and load timeline to show recent stored readings.</p>
      )}
    </div>
  );
}

function getRiskStatus(value) {
  if (value === null) {
    return {
      label: "Pending",
      tone: "neutral",
      message: "Calculate risk to classify this patient into a clear action band."
    };
  }
  if (value > 70) {
    return {
      label: "High",
      tone: "high",
      message: "Needs urgent clinical review and safety planning."
    };
  }
  if (value > 40) {
    return {
      label: "Moderate",
      tone: "moderate",
      message: "Monitor closely and consider a follow-up care plan."
    };
  }
  return {
    label: "Low",
    tone: "low",
    message: "Continue routine monitoring and supportive guidance."
  };
}
