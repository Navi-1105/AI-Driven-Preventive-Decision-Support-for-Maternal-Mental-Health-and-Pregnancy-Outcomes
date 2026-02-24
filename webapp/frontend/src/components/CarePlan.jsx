/**
 * CarePlan.jsx
 * RAG-Generated Care Plan Checklist Component
 * Displays evidence-based recommendations with source verification
 * Paper Section: III-F. RAG Advisory
 */

export default function CarePlan({
  items = [],
  sources = [],
  onToggleItem = null,
  onVerifySource = null,
  title = "Evidence-Based Care Plan"
}) {
  if (!items || items.length === 0) {
    return (
      <div className="care-plan empty">
        <h3>{title}</h3>
        <div className="care-plan-empty-message">
          <p className="muted">No care plan items available yet.</p>
          <p className="muted instruction-text">
            💡 <strong>How to get recommendations:</strong>
          </p>
          <ul className="instruction-list">
            <li>Submit a patient message in the chat</li>
            <li>Or run a risk assessment</li>
            <li>Recommendations will appear here automatically</li>
          </ul>
        </div>
      </div>
    );
  }

  const checkedCount = items.filter((item) => {
    const isChecked = item.checked || false;
    return isChecked;
  }).length;

  return (
    <div className="care-plan">
      <div className="care-plan-header">
        <h3>{title}</h3>
        <p className="care-plan-instruction">
          Review and mark completed recommendations. Click "Verify Source" to view evidence-based guidelines.
        </p>
        {sources && sources.length > 0 && (
          <div className="care-plan-sources">
            <span className="muted">Evidence Sources: </span>
            {sources.map((source, idx) => (
              <button
                key={`source-${idx}`}
                className="chip chip-source"
                onClick={() => {
                  if (onVerifySource) {
                    onVerifySource(source);
                  }
                }}
                title={`Click to verify source: ${source}`}
              >
                {source.split(" (")[0]}
              </button>
            ))}
          </div>
        )}
      </div>

      <div className="care-plan-items">
        {items.map((item, idx) => {
          const itemId = item.id || `item-${idx}`;
          const isChecked = item.checked || false;
          const itemText = typeof item === "string" ? item : item.text || item;

          return (
            <div key={itemId} className={`care-plan-item ${isChecked ? "checked" : ""}`}>
              <label className="care-plan-checkbox">
                <input
                  type="checkbox"
                  checked={isChecked}
                  onChange={() => {
                    if (onToggleItem) {
                      onToggleItem(itemId);
                    }
                  }}
                  disabled={!onToggleItem}
                  aria-label={`Mark "${itemText.substring(0, 30)}..." as ${isChecked ? "incomplete" : "completed"}`}
                />
                <span className="care-plan-text">{itemText}</span>
              </label>
              {(item.source || (sources && sources.length > 0)) && (
                <button
                  className="care-plan-verify"
                  onClick={() => {
                    if (onVerifySource) {
                      onVerifySource(item.source || sources[0]);
                    }
                  }}
                  title="Click to view evidence source"
                >
                  Verify Source
                </button>
              )}
            </div>
          );
        })}
      </div>

      {checkedCount > 0 && (
        <div className="care-plan-summary">
          <div className="summary-progress">
            <div className="progress-bar">
              <div 
                className="progress-fill" 
                style={{ width: `${(checkedCount / items.length) * 100}%` }}
              />
            </div>
            <p className="muted">
              <strong>{checkedCount}</strong> of <strong>{items.length}</strong> recommendations completed
            </p>
          </div>
        </div>
      )}
    </div>
  );
}
