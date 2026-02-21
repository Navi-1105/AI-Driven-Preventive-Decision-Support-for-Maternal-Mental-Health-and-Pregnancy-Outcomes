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
        <p className="muted">No care plan items available. Generate guidance to see recommendations.</p>
      </div>
    );
  }

  return (
    <div className="care-plan">
      <div className="care-plan-header">
        <h3>{title}</h3>
        {sources && sources.length > 0 && (
          <div className="care-plan-sources">
            <span className="muted">Sources: </span>
            {sources.map((source, idx) => (
              <button
                key={`source-${idx}`}
                className="chip chip-source"
                onClick={() => onVerifySource && onVerifySource(source)}
                title={`Verify source: ${source}`}
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
                  onChange={() => onToggleItem && onToggleItem(itemId)}
                  disabled={!onToggleItem}
                />
                <span className="care-plan-text">{itemText}</span>
              </label>
              {item.source && (
                <button
                  className="care-plan-verify"
                  onClick={() => onVerifySource && onVerifySource(item.source)}
                >
                  Verify Source
                </button>
              )}
            </div>
          );
        })}
      </div>

      {items.filter((item) => item.checked).length > 0 && (
        <div className="care-plan-summary">
          <p className="muted">
            {items.filter((item) => item.checked).length} of {items.length} items completed
          </p>
        </div>
      )}
    </div>
  );
}
