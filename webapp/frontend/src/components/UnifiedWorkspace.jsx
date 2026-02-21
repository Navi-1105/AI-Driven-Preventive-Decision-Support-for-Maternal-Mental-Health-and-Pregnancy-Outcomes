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
  risk,
  timeline,
  handleRisk,
  handleTimeline,

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
      {/* Left Column: Patient Profile & Context */}
      <div className="workspace-column left-column">
        <div className="column-card">
          <h2>Patient Profile & Context</h2>

          {/* Patient Identity */}
          <div className="patient-identity-section">
            <h3>Patient Information</h3>
            <div className="identity-grid">
              <label>
                Patient ID
                <input
                  value={inputs.patient_id}
                  onChange={(e) => updateInput("patient_id", e.target.value)}
                  placeholder="patient-001"
                />
              </label>
              <label>
                Patient Name
                <input
                  value={patientIdentity.patient_name}
                  onChange={(e) => setPatientIdentity((p) => ({ ...p, patient_name: e.target.value }))}
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
                  onChange={(e) => updateInput("age", e.target.value)}
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
              <button onClick={handleRisk}>Calculate Risk</button>
              <button className="secondary" onClick={handleTimeline}>Load Timeline</button>
            </div>

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
          </div>
        </div>
      </div>

      {/* Middle Column: Interactive Smart Chat */}
      <div className="workspace-column middle-column">
        <div className="column-card">
          <div className="chat-header">
            <h2>Clinical Co-Pilot Chat</h2>
            {chatResult && (
              <div className="chat-risk-indicator">
                <RiskBadge value={chatResult.risk_percent} />
              </div>
            )}
          </div>

          {/* Crisis Banner */}
          {(chatResult?.crisis_mode || risk?.crisis_mode) && (
            <SafetyOverride
              crisisMode={true}
              onEscalate={handleEscalate}
              riskPercent={chatResult?.risk_percent || risk?.risk_percent}
            />
          )}

          {/* Chat Input */}
          <div className="chat-input-section">
            <label>
              Patient Message
              <textarea
                rows="4"
                value={chatMessage}
                onChange={(e) => setChatMessage(e.target.value)}
                placeholder="Describe current emotional state, sleep, appetite, stress, and postpartum concerns..."
                disabled={chatLocked}
              />
            </label>
            <div className="row">
              <button onClick={handleChatAssess} disabled={chatLoading || chatLocked}>
                {chatLoading ? "Analyzing..." : "Analyze Message"}
              </button>
              <label className="file-upload-label">
                <input
                  type="file"
                  accept=".txt,.md,.csv"
                  onChange={handleTranscriptUpload}
                  disabled={chatLocked}
                  style={{ display: "none" }}
                />
                <span className="secondary">Upload Transcript</span>
              </label>
            </div>
          </div>

          {/* Chat Thread */}
          <div className="chat-thread-container">
            <h3>Conversation Thread</h3>
            {chatHistory && chatHistory.length > 0 ? (
              <div className="chat-thread">
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
              <p className="muted">No messages yet. Submit a patient message to start the conversation.</p>
            )}
          </div>
        </div>
      </div>

      {/* Right Column: Clinical Insights & RAG Care Plan */}
      <div className="workspace-column right-column">
        {/* SHAP XAI Inspector */}
        <div className="column-card">
          <ShapExplanation
            contributions={xai?.contributions || []}
            riskPercent={risk?.risk_percent || chatResult?.risk_percent}
            crisisMode={risk?.crisis_mode || chatResult?.crisis_mode}
          />
        </div>

        {/* RAG Care Plan */}
        <div className="column-card">
          <CarePlan
            items={carePlanItems || []}
            sources={chatResult?.sources || ragResponse?.sources || []}
            onToggleItem={toggleCarePlanItem}
            onVerifySource={onOpenSource}
            title="Evidence-Based Care Plan"
          />
        </div>
      </div>
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

// Enhanced Smart Chat Bubble with Keyword Highlighting
function SmartChatBubble({ item, itemState, onToggleCarePlan, onReview, onOpenSource, xaiContributions }) {
  const isUser = item.role === "user";
  const isSystem = item.role === "system";
  const isAssistant = item.role === "assistant";
  const review = itemState?.review || "";

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
      <p className="bubble-role">{isUser ? "Patient" : "AI Clinical Assistant"}</p>
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

      {/* Assistant Actions */}
      {isAssistant && (
        <div className="bubble-actions">
          <button
            className={review === "agree" ? "" : "secondary"}
            onClick={() => onReview(item.id, "agree")}
          >
            ✓ Agree
          </button>
          <button
            className={review === "correct" ? "" : "secondary"}
            onClick={() => onReview(item.id, "correct")}
          >
            ✎ Correct
          </button>
        </div>
      )}
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
