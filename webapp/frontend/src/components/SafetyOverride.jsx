/**
 * SafetyOverride.jsx
 * Red Flag Crisis Banner Component
 * Displays persistent crisis alert when self-harm indicators are detected
 * Paper Section: III-E. Red Flag Protocol
 */

export default function SafetyOverride({ crisisMode = false, onEscalate = null, riskPercent = null }) {
  if (!crisisMode) {
    return null;
  }

  return (
    <div className="safety-override">
      <div className="crisis-header">
        <div className="crisis-icon">⚠️</div>
        <div className="crisis-content">
          <h3 className="crisis-title">CRISIS PROTOCOL ACTIVATED</h3>
          <p className="crisis-message">
            Self-harm indicator detected. Immediate clinical intervention required.
            {riskPercent !== null && (
              <span className="crisis-risk"> Risk Level: {riskPercent.toFixed(1)}%</span>
            )}
          </p>
        </div>
      </div>

      <div className="crisis-actions">
        {onEscalate && (
          <>
            <button
              className="crisis-btn emergency"
              onClick={() => onEscalate("emergency_services_contacted")}
            >
              🚨 Contact Emergency Services
            </button>
            <button
              className="crisis-btn urgent"
              onClick={() => onEscalate("crisis_team_notified")}
            >
              📞 Notify Crisis Team
            </button>
            <a className="crisis-btn link" href="tel:988">
              📱 Call 988 Suicide & Crisis Lifeline
            </a>
          </>
        )}
      </div>

      <div className="crisis-footer">
        <p className="muted">
          <strong>Note:</strong> This alert persists until manually dismissed. All AI recommendations are overridden
          during crisis mode.
        </p>
      </div>
    </div>
  );
}
