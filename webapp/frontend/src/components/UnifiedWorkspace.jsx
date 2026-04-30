/**
 * UnifiedWorkspace.jsx
 * 3-Column Unified Patient Workspace
 * Left: Patient Profile & Context
 * Middle: Interactive Smart Chat
 * Right: Clinical Insights & RAG Care Plan
 */

import RiskTrend from "./RiskTrend";
import ShapExplanation from "./ShapExplanation";
import SafetyOverride from "./SafetyOverride";
import CarePlan from "./CarePlan";

export default function UnifiedWorkspace({
  // Patient Profile & Context (Left Column)
  inputs,
  updateInput,
  patientIdentity,
  setPatientIdentity,
  patientStatus,
  patientStatusClass,
  clinicalActionsEnabled,
  timelineActionsEnabled,
  timelineStatus,
  risk,
  timeline,
  handleRisk,
  handleTimeline,
  handlePatientBlur,
  handleSavePatient,

  // Smart Chat (Middle Column)
  chatMessage,
  setChatMessage,
  chatHistory,
  chatItemState,
  chatResult,
  chatLoading,
  chatLocked,
  handleChatAssess,
  handleEscalate,
  handleTranscriptUpload,
  toggleChatCarePlanItem,
  setChatMessageReview,
  onOpenSource,

  // Clinical Insights (Right Column)
  xai,
  xaiStatus,
  carePlanItems,
  setCarePlanItems,
  toggleCarePlanItem,
  ragResponse,
  chatResultSources
}) {
  return (
    <div className="unified-workspace">
      <aside className="workspace-sidebar workspace-sidebar-left">
        <div className="column-card">
          <div className="section-header">
            <h2>Patient Context</h2>
            <p className="section-description">
              Clinical profile, dynamic factors, and timeline context for the active conversation.
            </p>
          </div>

          {/* Patient Identity */}
          <div className="patient-identity-section">
            <h3>Patient Information</h3>
            <div className="identity-grid">
              <label>
                Patient ID
                <input
                  value={inputs.patient_id}
                  onChange={(e) => updateInput("patient_id", e.target.value)}
                  onBlur={handlePatientBlur}
                  placeholder="patient-001"
                />
              </label>
              <label>
                Patient Name
                <input
                  value={patientIdentity.name}
                  onChange={(e) => setPatientIdentity((p) => ({ ...p, name: e.target.value }))}
                  placeholder="Full Name"
                />
              </label>
              <label>
                DOB
                <input
                  type="date"
                  value={patientIdentity.dob}
                  onChange={(e) => setPatientIdentity((p) => ({ ...p, dob: e.target.value }))}
                />
              </label>
              <label>
                MRN
                <input
                  value={patientIdentity.mrn}
                  onChange={(e) => setPatientIdentity((p) => ({ ...p, mrn: e.target.value }))}
                  placeholder="Medical Record Number"
                />
              </label>
            </div>
            <div className="row">
              <button className="secondary" type="button" onClick={handleSavePatient}>
                Save Patient
              </button>
              {patientStatus ? (
                <span className={`muted ${patientStatusClass || ""}`}>
                  {patientStatus}
                </span>
              ) : null}
            </div>
          </div>

          {/* Dynamic Risk Profiler Inputs */}
          <div className="risk-profiler-section">
            <h3>Dynamic Risk Profiler</h3>
            <div className="form-grid">
              <label>
                Gestational Weeks
                <input
                  type="number"
                  value={inputs.gestational_weeks}
                  onChange={(e) => updateInput("gestational_weeks", e.target.value)}
                  min="0"
                  max="42"
                />
              </label>
              <label>
                Age
                <input
                  type="number"
                  value={inputs.age}
                  readOnly
                  title="Age is calculated from patient DOB"
                  min="15"
                  max="50"
                />
              </label>
              <label>
                Income Band
                <select
                  value={inputs.income_band}
                  onChange={(e) => updateInput("income_band", e.target.value)}
                >
                  <option value="low">Low</option>
                  <option value="middle">Middle</option>
                  <option value="high">High</option>
                </select>
              </label>
              <label>
                Sleep Quality (0-10)
                <input
                  type="number"
                  value={inputs.sleep_quality}
                  onChange={(e) => updateInput("sleep_quality", e.target.value)}
                  min="0"
                  max="10"
                />
              </label>
              <label>
                Appetite (0-10)
                <input
                  type="number"
                  value={inputs.appetite}
                  onChange={(e) => updateInput("appetite", e.target.value)}
                  min="0"
                  max="10"
                />
              </label>
              <label>
                Fatigue (0-10)
                <input
                  type="number"
                  value={inputs.fatigue}
                  onChange={(e) => updateInput("fatigue", e.target.value)}
                  min="0"
                  max="10"
                />
              </label>
              <label>
                Financial Stress (0-10)
                <input
                  type="number"
                  value={inputs.financial_stress}
                  onChange={(e) => updateInput("financial_stress", e.target.value)}
                  min="0"
                  max="10"
                />
              </label>
              <label>
                Self-harm Thoughts (0-5)
                <input
                  type="number"
                  value={inputs.self_harm}
                  onChange={(e) => updateInput("self_harm", e.target.value)}
                  min="0"
                  max="5"
                />
              </label>
            </div>
            <div className="row">
              <button onClick={handleRisk} disabled={!clinicalActionsEnabled}>Calculate Risk</button>
              <button className="secondary" onClick={handleTimeline} disabled={!timelineActionsEnabled}>Load Timeline</button>
            </div>
            {timelineStatus ? <p className="muted">{timelineStatus}</p> : null}

            {/* Current Risk Display */}
            {risk && typeof risk.risk_percent === "number" && (
              <div className={`risk-display ${risk.crisis_mode ? "crisis" : ""}`}>
                <div className="risk-badge-container">
                  <RiskBadge value={risk.risk_percent} />
                </div>
                <p className="muted">{risk.message}</p>
              </div>
            )}
          </div>

          {/* Risk Trend Chart */}
          <div className="risk-trend-section">
            <RiskTrend
              timelinePoints={timeline}
              currentRisk={risk?.risk_percent}
              gestationalWeeks={inputs.gestational_weeks}
            />
            <TimelineHistory points={timeline} />
          </div>
        </div>
      </aside>

      <main className="workspace-chat-shell">
        <div className="chat-shell-card">
          <div className="chat-shell-topbar">
            <div>
              <p className="chat-kicker">AI maternal mental health copilot</p>
              <h2>Clinical Co-Pilot Chat</h2>
            </div>
            <div className="chat-topbar-meta">
              {chatResult && (
                <div className="chat-risk-indicator">
                  <RiskBadge value={chatResult.risk_percent} />
                </div>
              )}
              <span className={`chat-status-pill ${chatLocked ? "locked" : "live"}`}>
                {chatLocked ? "Crisis lock active" : "Ready for triage"}
              </span>
            </div>
          </div>

          {!chatHistory?.length ? (
            <div className="chat-welcome">
              <div className="chat-welcome-copy">
                <h3>Start a safer, guided patient conversation</h3>
                <p className="section-description">
                  Enter patient text or upload a transcript. Risk signals, evidence highlights, and suggested actions appear inline as the thread builds.
                </p>
              </div>
              <div className="chat-suggestion-row">
                {[
                  "I have not been sleeping since delivery and feel overwhelmed.",
                  "I feel constantly anxious and cannot stop crying.",
                  "Please review this postpartum follow-up transcript."
                ].map((suggestion) => (
                  <button
                    key={suggestion}
                    type="button"
                    className="prompt-chip"
                    onClick={() => setChatMessage(suggestion)}
                    disabled={chatLocked}
                  >
                    {suggestion}
                  </button>
                ))}
              </div>
            </div>
          ) : null}

          {(chatResult?.crisis_mode || risk?.crisis_mode) && (
            <SafetyOverride
              crisisMode={true}
              onEscalate={handleEscalate}
              riskPercent={chatResult?.risk_percent || risk?.risk_percent}
            />
          )}

          <div className="chat-thread-stage">
            {chatHistory && chatHistory.length > 0 ? (
              <div className="chat-thread chat-thread-shell">
                {chatHistory.map((item, index) => (
                  <SmartChatBubble
                    key={`${item.id || item.at}-${index}`}
                    item={item}
                    itemState={chatItemState[item.id] || { checked: {}, review: "" }}
                    onToggleCarePlan={toggleChatCarePlanItem}
                    onReview={setChatMessageReview}
                    onOpenSource={onOpenSource}
                    xaiContributions={xai?.contributions}
                  />
                ))}
              </div>
            ) : (
              <div className="chat-empty-state">
                <p>No messages yet.</p>
                <span className="muted">Submit a patient message to start the thread.</span>
              </div>
            )}
          </div>

          <div className="chat-composer">
            <label className="composer-field">
              <span className="sr-only">Patient Message</span>
              <textarea
                rows="3"
                value={chatMessage}
                onChange={(e) => setChatMessage(e.target.value)}
                placeholder="Message the clinical copilot with patient symptoms, mood shifts, sleep issues, or uploaded transcript notes..."
                disabled={chatLocked}
              />
            </label>
            <div className="composer-actions">
              <label className="upload-pill">
                <input
                  type="file"
                  accept=".txt,.md,.csv,.pdf"
                  onChange={handleTranscriptUpload}
                  disabled={chatLocked}
                  style={{ display: "none" }}
                  title="Supported formats: Text (.txt), Markdown (.md), CSV (.csv), PDF (.pdf)"
                />
                Attach transcript
              </label>
              <span className="muted">.txt .md .csv .pdf</span>
              <button onClick={handleChatAssess} disabled={chatLoading || chatLocked || !clinicalActionsEnabled}>
                {chatLoading ? "Analyzing..." : "Send for analysis"}
              </button>
            </div>
          </div>
        </div>
      </main>

      <aside className="workspace-sidebar workspace-sidebar-right">
        <div className="column-card">
          <div className="section-header">
            <h2>Feature Impact Analysis</h2>
            <p className="section-description">
              Understand which factors contribute most to the risk score. Positive values increase risk, negative values decrease risk.
            </p>
          </div>
          <ShapExplanation
            contributions={xai?.contributions || []}
            riskPercent={risk?.risk_percent || chatResult?.risk_percent}
            crisisMode={risk?.crisis_mode || chatResult?.crisis_mode}
          />
        </div>

        {/* RAG Care Plan */}
        <div className="column-card">
          <div className="section-header">
            <h2>Recommended Actions</h2>
            <p className="section-description">
              Evidence-based care plan generated from clinical guidelines. Mark items as completed and verify sources for transparency.
            </p>
          </div>
          <CarePlan
            items={carePlanItems || []}
            sources={chatResult?.sources || ragResponse?.sources || []}
            onToggleItem={(itemId) => {
              if (toggleCarePlanItem && itemId) {
                toggleCarePlanItem(itemId);
              }
            }}
            onVerifySource={(source) => {
              if (onOpenSource && source) {
                onOpenSource(source);
              }
            }}
            title="Evidence-Based Care Plan"
          />
        </div>
      </aside>
    </div>
  );
}

// Risk Badge Component (Pill-style)
function RiskBadge({ value }) {
  const safeValue = Math.max(0, Math.min(100, Number(value || 0)));
  let level = "low";
  let className = "risk-badge low";

  if (safeValue > 70) {
    level = "high";
    className = "risk-badge high";
  } else if (safeValue > 40) {
    level = "moderate";
    className = "risk-badge moderate";
  }

  return (
    <div className={className}>
      <span className="risk-badge-label">{level.toUpperCase()}</span>
      <span className="risk-badge-value">{safeValue.toFixed(1)}%</span>
    </div>
  );
}

function TimelineHistory({ points = [] }) {
  const rows = [...points].sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp));
  if (!rows.length) {
    return <p className="muted">No stored timeline entries yet.</p>;
  }

  return (
    <div className="timeline-history">
      <h3>Timeline History</h3>
      <div className="timeline-table">
        {rows.map((row, index) => (
          <div className="timeline-row" key={`${row.timestamp}-${index}`}>
            <span>
              Week {row.gestational_weeks}
              <small>{formatTimelineDate(row.timestamp)}</small>
            </span>
            <strong>{Number(row.risk_percent).toFixed(1)}%</strong>
          </div>
        ))}
      </div>
    </div>
  );
}

function formatTimelineDate(timestamp) {
  const date = new Date(timestamp);
  if (Number.isNaN(date.getTime())) return "";
  return date.toLocaleString([], {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit"
  });
}

// Enhanced Smart Chat Bubble with Keyword Highlighting
function SmartChatBubble({ item, itemState, onToggleCarePlan, onReview, onOpenSource, xaiContributions }) {
  const isUser = item.role === "user";
  const isSystem = item.role === "system";
  const isAssistant = item.role === "assistant";

  if (isSystem) {
    return (
      <div className="bubble system">
        <p className="bubble-role">System Alert</p>
        <p className="bubble-text">{item.text}</p>
      </div>
    );
  }

  // Enhanced keyword highlighting with SHAP integration
  const highlightedText = isUser
    ? highlightKeywordsWithSHAP(item.text, item.highlightFactors || [], xaiContributions)
    : item.text;

  return (
    <div className={`bubble ${isUser ? "user" : "assistant"}`}>
      <div className="bubble-heading">
        <p className="bubble-role">{isUser ? "Patient" : "AI Clinical Assistant"}</p>
        {item.at ? (
          <span className="bubble-time">
            {new Date(item.at).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
          </span>
        ) : null}
      </div>
      <div className="bubble-text">{highlightedText}</div>

      {item.meta?.risk_percent !== undefined && (
        <div className="bubble-meta">
          <RiskBadge value={item.meta.risk_percent} />
          <span className="muted">Context: {item.meta.likely_context}</span>
        </div>
      )}

      {/* Integrated RAG Cards */}
      {isAssistant && item.carePlan?.length ? (
        <div className="bubble-careplan">
          <h4 className="careplan-title">Recommended Actions</h4>
          {item.carePlan.map((plan) => (
            <div key={plan.id} className="rag-card">
              <label className="rag-card-checkbox">
                <input
                  type="checkbox"
                  checked={Boolean(itemState?.checked?.[plan.id])}
                  onChange={() => onToggleCarePlan(item.id, plan.id)}
                />
                <span className="rag-card-text">{plan.text}</span>
              </label>
              {item.meta?.sources?.[0] && (
                <button
                  className="rag-card-verify"
                  onClick={() => onOpenSource(item.meta.sources[0])}
                >
                  Verify Source
                </button>
              )}
            </div>
          ))}
        </div>
      ) : null}

      {isAssistant && item.meta?.sources?.length ? (
        <div className="bubble-actions">
          {item.meta.sources.map((source) => (
            <button
              key={`${item.id}-${source}`}
              className="chip chip-source"
              onClick={() => onOpenSource(source)}
              title={source}
            >
              Source: {sourceLabelToName(source)}
            </button>
          ))}
        </div>
      ) : null}
    </div>
  );
}

// Enhanced keyword highlighting with SHAP contribution tooltips
function highlightKeywordsWithSHAP(text, factors, contributions = []) {
  const keywordsMap = {
    sleep_disturbance: ["sleep", "insomnia", "neend", "sleepless", "trouble sleeping"],
    low_mood: ["sad", "hopeless", "udaas", "depressed", "down", "blue"],
    anxiety: ["anxious", "panic", "chinta", "tension", "worried", "nervous"],
    fatigue: ["fatigue", "tired", "exhausted", "thakan", "worn out", "drained"],
    appetite_change: ["appetite", "bhook", "eating", "hunger", "food"],
    self_harm_ideation: [
      "hurt myself",
      "kill myself",
      "better off dead",
      "jeene ka mann nahi",
      "suicide",
      "end it all"
    ]
  };

  // Find SHAP contribution for each factor
  const getSHAPContribution = (factor) => {
    const contrib = contributions.find((c) => c.feature === factor);
    return contrib ? contrib.contribution_percent : null;
  };

  let highlighted = text;
  factors.forEach((factor) => {
    const shapValue = getSHAPContribution(factor);
    const impact = shapValue !== null ? shapValue : estimateFactorImpact(factor);
    const keywords = keywordsMap[factor] || [];

    keywords.forEach((keyword) => {
      const regex = new RegExp(`\\b(${escapeRegExp(keyword)})\\b`, "gi");
      highlighted = highlighted.replace(
        regex,
        `<mark class="evidence-mark shap-highlight" data-factor="${factor}" data-impact="${impact.toFixed(1)}" title="${factor.replace(/_/g, " ")} contributed ${impact > 0 ? "+" : ""}${impact.toFixed(1)}% to risk score">$1</mark>`
      );
    });
  });

  return <span dangerouslySetInnerHTML={{ __html: highlighted }} />;
}

function escapeRegExp(value) {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function sourceLabelToName(label) {
  return (label || "").split(" (score=")[0].trim();
}

function estimateFactorImpact(factor) {
  const weights = {
    sleep_disturbance: 15,
    low_mood: 18,
    anxiety: 12,
    fatigue: 11,
    appetite_change: 8,
    self_harm_ideation: 35
  };
  return weights[factor] || 5;
}
