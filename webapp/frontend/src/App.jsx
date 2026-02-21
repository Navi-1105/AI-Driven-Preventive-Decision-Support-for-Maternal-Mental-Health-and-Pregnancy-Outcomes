import { useEffect, useMemo, useState } from "react";
import "./styles.css";
import UnifiedWorkspace from "./components/UnifiedWorkspace";

const API_BASE = import.meta.env.VITE_API_BASE || "http://127.0.0.1:8000";
const STORAGE_KEYS = {
  token: "ppds_token",
  role: "ppds_role",
  username: "ppds_username",
  patientId: "ppds_patient_id"
};

const initialInputs = {
  patient_id: "",
  gestational_weeks: 20,
  sleep_quality: 6,
  appetite: 6,
  fatigue: 5,
  financial_stress: 4,
  self_harm: 0,
  age: 28,
  income_band: "middle"
};

const NAV_ITEMS = [
  { key: "overview", label: "Overview" },
  { key: "workspace", label: "Unified Workspace" },
  { key: "chat", label: "Chat Triage" },
  { key: "clinical", label: "Clinical Review", roles: ["clinician", "admin"] },
  { key: "risk", label: "Risk & Timeline" },
  { key: "guidance", label: "Guidance & Ethics" }
];

export default function App() {
  const [auth, setAuth] = useState({
    username: getStoredValue(STORAGE_KEYS.username, "clin1"),
    password: "StrongPass123",
    role: "clinician"
  });
  const [token, setToken] = useState(() => getStoredValue(STORAGE_KEYS.token, ""));
  const [currentRole, setCurrentRole] = useState(() => getStoredValue(STORAGE_KEYS.role, ""));
  const [activePage, setActivePage] = useState(() => {
    const role = getStoredValue(STORAGE_KEYS.role, "");
    return role === "clinician" || role === "admin" ? "workspace" : "overview";
  });

  const [apiStatus, setApiStatus] = useState("");
  const [backendHealth, setBackendHealth] = useState({ status: "checking", message: "Checking backend..." });
  const [authLoading, setAuthLoading] = useState(false);
  const [authStatus, setAuthStatus] = useState("");

  const [inputs, setInputs] = useState(() => ({
    ...initialInputs,
    patient_id: getStoredValue(STORAGE_KEYS.patientId, "")
  }));
  const [patientIdentity, setPatientIdentity] = useState({
    patient_name: "",
    dob: "",
    mrn: ""
  });
  const [consentStatus, setConsentStatus] = useState("");

  const [risk, setRisk] = useState(null);
  const [xai, setXai] = useState(null);
  const [xaiStatus, setXaiStatus] = useState("");
  const [timeline, setTimeline] = useState([]);

  const [chatMessage, setChatMessage] = useState(
    "I gave birth 6 weeks ago, I feel exhausted, cannot sleep, and feel like a bad mother."
  );
  const [chatResult, setChatResult] = useState(null);
  const [chatHistory, setChatHistory] = useState([]);
  const [chatItemState, setChatItemState] = useState({});
  const [caseStatus, setCaseStatus] = useState("new");
  const [chatLoading, setChatLoading] = useState(false);
  const [chatLocked, setChatLocked] = useState(false);
  const [carePlanItems, setCarePlanItems] = useState([]);
  const [sourcePreview, setSourcePreview] = useState(null);
  const [escalationStatus, setEscalationStatus] = useState("");

  const [ragQuery, setRagQuery] = useState("Safe sleep tips third trimester");
  const [ragDrivers, setRagDrivers] = useState("sleep disturbance, fatigue");
  const [ragResponse, setRagResponse] = useState(null);

  const [fairnessGroups, setFairnessGroups] = useState([
    { group: "low_income", positive_rate: 0.25 },
    { group: "high_income", positive_rate: 0.35 }
  ]);
  const [fairnessResult, setFairnessResult] = useState(null);

  const [feedbackLabel, setFeedbackLabel] = useState("low");
  const [ehrSummary, setEhrSummary] = useState(null);
  const [outcomeLabel, setOutcomeLabel] = useState("stable");
  const [outcomeNotes, setOutcomeNotes] = useState("");

  const role = currentRole || auth.role;
  const timelinePoints = useMemo(() => timeline, [timeline]);

  const visibleNav = NAV_ITEMS.filter((item) => !item.roles || item.roles.includes(role));

  useEffect(() => {
    if (!visibleNav.some((item) => item.key === activePage)) {
      setActivePage("overview");
    }
  }, [role]);

  useEffect(() => {
    setStoredValue(STORAGE_KEYS.token, token);
  }, [token]);

  useEffect(() => {
    setStoredValue(STORAGE_KEYS.role, currentRole);
  }, [currentRole]);

  useEffect(() => {
    setStoredValue(STORAGE_KEYS.username, auth.username);
  }, [auth.username]);

  useEffect(() => {
    setStoredValue(STORAGE_KEYS.patientId, inputs.patient_id);
  }, [inputs.patient_id]);

  const updateInput = (field, value) => {
    setInputs((prev) => ({ ...prev, [field]: value }));
  };

  const authHeaders = () => ({
    "Content-Type": "application/json",
    ...(token ? { Authorization: `Bearer ${token}` } : {})
  });

  const parseJsonSafe = async (response) => {
    try {
      return await response.json();
    } catch {
      return {};
    }
  };

  const requestJson = async (url, options = {}) => {
    let lastError = null;
    for (let attempt = 0; attempt < 2; attempt += 1) {
      const controller = new AbortController();
      const timeout = setTimeout(() => controller.abort(), 15000);
      try {
        const response = await fetch(url, { ...options, signal: controller.signal });
        const data = await parseJsonSafe(response);
        if (!response.ok) {
          const message = data?.detail || `Request failed (${response.status})`;
          throw new Error(message);
        }
        return data;
      } catch (error) {
        lastError = error;
        const isRetryable = error?.name === "AbortError" || error instanceof TypeError;
        if (!isRetryable || attempt === 1) {
          if (error?.name === "AbortError") {
            throw new Error(`Backend timeout at ${API_BASE}`);
          }
          throw error;
        }
      } finally {
        clearTimeout(timeout);
      }
    }
    throw lastError;
  };

  useEffect(() => {
    const checkHealth = async () => {
      const controller = new AbortController();
      const timeout = setTimeout(() => controller.abort(), 2500);
      try {
        const response = await fetch(`${API_BASE}/`, { signal: controller.signal });
        if (!response.ok) {
          throw new Error(`Health check failed (${response.status})`);
        }
        const data = await parseJsonSafe(response);
        setBackendHealth({ status: "online", message: `Backend online (${data.app || "service"})` });
      } catch {
        setBackendHealth({
          status: "offline",
          message: `Backend offline at ${API_BASE}. Start backend or set VITE_API_BASE.`
        });
      } finally {
        clearTimeout(timeout);
      }
    };

    checkHealth();
  }, []);

  useEffect(() => {
    let cancelled = false;

    const checkConsentStatus = async () => {
      if (!token) {
        if (!cancelled) setConsentStatus("");
        return;
      }
      if (!inputs.patient_id) {
        if (!cancelled) setConsentStatus("Enter patient ID to check consent");
        return;
      }

      try {
        const data = await requestJson(`${API_BASE}/api/privacy/consent/${encodeURIComponent(inputs.patient_id)}`, {
          headers: authHeaders()
        });
        if (cancelled) return;
        if (data?.consent_given) {
          setConsentStatus(`Consent active (${(data.consent_scope || []).join(", ")})`);
        } else {
          setConsentStatus("Consent not granted");
        }
      } catch (error) {
        if (cancelled) return;
        if ((error?.message || "").toLowerCase().includes("not found")) {
          setConsentStatus("Consent not granted");
        } else {
          setConsentStatus(`Consent check failed: ${error.message}`);
        }
      }
    };

    checkConsentStatus();
    return () => {
      cancelled = true;
    };
  }, [token, inputs.patient_id]);

  const buildPayload = () => ({
    patient_id: inputs.patient_id || undefined,
    gestational_weeks: Number(inputs.gestational_weeks),
    behavioral: {
      sleep_quality: Number(inputs.sleep_quality),
      appetite: Number(inputs.appetite),
      fatigue: Number(inputs.fatigue),
      financial_stress: Number(inputs.financial_stress)
    },
    demographics: {
      age: Number(inputs.age),
      income_band: inputs.income_band
    },
    self_harm: Number(inputs.self_harm)
  });

  const handleRegister = async () => {
    try {
      setAuthLoading(true);
      setApiStatus("");
      const data = await requestJson(`${API_BASE}/api/auth/register`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(auth)
      });
      setToken(data.access_token);
      setCurrentRole(data.role);
      setAuthStatus(`Registered as ${data.role}`);
      setActivePage(data.role === "clinician" || data.role === "admin" ? "workspace" : "overview");
    } catch (error) {
      setAuthStatus(error.message || "Register failed");
    } finally {
      setAuthLoading(false);
    }
  };

  const handleLogin = async () => {
    try {
      setAuthLoading(true);
      setApiStatus("");
      const data = await requestJson(`${API_BASE}/api/auth/login`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ username: auth.username, password: auth.password })
      });
      setToken(data.access_token);
      setCurrentRole(data.role);
      setAuthStatus(`Logged in as ${data.role}`);
      setActivePage(data.role === "clinician" || data.role === "admin" ? "workspace" : "overview");
    } catch (error) {
      setAuthStatus(error.message || "Login failed");
    } finally {
      setAuthLoading(false);
    }
  };

  const handleLogout = () => {
    setToken("");
    setCurrentRole("");
    setActivePage("overview");
    setApiStatus("");
    setAuthStatus("");
    setConsentStatus("");
    setRisk(null);
    setXai(null);
    setTimeline([]);
    setChatResult(null);
    setChatHistory([]);
    setChatItemState({});
    setChatLocked(false);
    setRagResponse(null);
    setFairnessResult(null);
    clearStoredSession();
  };

  const handleConsent = async () => {
    try {
      if (!inputs.patient_id || !token) return;
      setApiStatus("");
      const data = await requestJson(`${API_BASE}/api/privacy/consent`, {
        method: "POST",
        headers: authHeaders(),
        body: JSON.stringify({
          patient_id: inputs.patient_id,
          consent_given: true,
          consent_scope: ["chat_assessment", "risk_scoring", "model_improvement"],
          updated_by: auth.username
        })
      });
      if (data.patient_id) {
        setConsentStatus(
          data.consent_given
            ? `Consent active (${(data.consent_scope || []).join(", ")})`
            : "Consent not granted"
        );
      }
    } catch (error) {
      setConsentStatus(error.message || "Consent failed");
      setApiStatus(error.message || "Consent failed");
    }
  };

  const handleRisk = async () => {
    try {
      setApiStatus("");
      const payload = buildPayload();
      const data = await requestJson(`${API_BASE}/api/risk`, {
        method: "POST",
        headers: authHeaders(),
        body: JSON.stringify(payload)
      });
      setRisk(data);

      const xaiData = await requestJson(`${API_BASE}/api/xai`, {
        method: "POST",
        headers: authHeaders(),
        body: JSON.stringify(payload)
      });
      setXai(xaiData);
      if (data.crisis_mode) {
        setXaiStatus("XAI hidden: crisis protocol override is active (self-harm signal detected).");
      } else if (xaiData?.contributions?.length) {
        setXaiStatus("");
      } else {
        setXaiStatus("No feature contributions returned for this record.");
      }
      if (inputs.patient_id) {
        const timelineData = await requestJson(`${API_BASE}/api/timeline/${inputs.patient_id}`, {
          headers: token ? { Authorization: `Bearer ${token}` } : {}
        });
        setTimeline(timelineData.points || []);
      }
    } catch (error) {
      setRisk(null);
      setXai(null);
      setXaiStatus("");
      setApiStatus(error.message);
    }
  };

  const handleTimeline = async () => {
    try {
      if (!inputs.patient_id) return;
      setApiStatus("");
      const data = await requestJson(`${API_BASE}/api/timeline/${inputs.patient_id}`, {
        headers: token ? { Authorization: `Bearer ${token}` } : {}
      });
      setTimeline(data.points || []);
    } catch (error) {
      setTimeline([]);
      setApiStatus(error.message);
    }
  };

  const handleChatAssess = async () => {
    if (chatLocked) return;
    setChatLoading(true);
    try {
      setApiStatus("");
      const runId = `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
      const userMessage = {
        id: `u-${runId}`,
        role: "user",
        text: chatMessage,
        at: new Date().toISOString()
      };
      const data = await requestJson(`${API_BASE}/api/chat-assess`, {
        method: "POST",
        headers: authHeaders(),
        body: JSON.stringify({ patient_id: inputs.patient_id || undefined, message: chatMessage })
      });
      const assistantId = `a-${runId}`;
      const carePlan = toKeyPoints(data.guidance).map((text, idx) => ({
        id: `${assistantId}-${idx}`,
        text
      }));
      setChatResult(data);
      setChatHistory((prev) => [
        ...prev,
        { ...userMessage, highlightFactors: data.risk_factors || [] },
        {
          id: assistantId,
          role: "assistant",
          text: data.guidance,
          at: new Date().toISOString(),
          meta: data,
          carePlan
        }
      ]);
      setChatItemState((prev) => ({
        ...prev,
        [assistantId]: {
          checked: {},
          review: ""
        }
      }));
      if (data.crisis_mode) {
        setChatLocked(true);
        setChatHistory((prev) => [
          ...prev,
          {
            id: `s-${runId}`,
            role: "system",
            text: "Crisis Protocol Triggered: On-call clinician notified.",
            at: new Date().toISOString()
          }
        ]);
      }
      setCaseStatus("pending_review");
      setCarePlanItems(carePlan.map((item) => ({ ...item, checked: false })));
    } catch (error) {
      setChatResult(null);
      setApiStatus(error.message);
    } finally {
      setChatLoading(false);
    }
  };

  const handleRag = async () => {
    try {
      setApiStatus("");
      const data = await requestJson(`${API_BASE}/api/rag`, {
        method: "POST",
        headers: authHeaders(),
        body: JSON.stringify({
          query: ragQuery,
          risk_drivers: ragDrivers
            .split(",")
            .map((d) => d.trim())
            .filter(Boolean)
        })
      });
      setRagResponse(data);
      setCarePlanItems(
        toKeyPoints(data.answer).map((text, idx) => ({ id: `${idx}-${text}`, text, checked: false }))
      );
    } catch (error) {
      setRagResponse(null);
      setApiStatus(error.message);
    }
  };

  const updateGroup = (index, field, value) => {
    setFairnessGroups((prev) =>
      prev.map((group, idx) => (idx === index ? { ...group, [field]: value } : group))
    );
  };

  const addGroup = () => {
    setFairnessGroups((prev) => [...prev, { group: "group", positive_rate: 0.2 }]);
  };

  const handleFairness = async () => {
    try {
      setApiStatus("");
      const data = await requestJson(`${API_BASE}/api/fairness`, {
        method: "POST",
        headers: authHeaders(),
        body: JSON.stringify({
          protected_attribute: "income_band",
          groups: fairnessGroups.map((g) => ({
            group: g.group,
            positive_rate: Number(g.positive_rate)
          }))
        })
      });
      setFairnessResult(data);
    } catch (error) {
      setFairnessResult(null);
      setApiStatus(error.message);
    }
  };

  const handleFeedback = async (label) => {
    if (!risk?.prediction_id) return;
    try {
      setApiStatus("");
      await requestJson(`${API_BASE}/api/feedback`, {
        method: "POST",
        headers: authHeaders(),
        body: JSON.stringify({ prediction_id: risk.prediction_id, clinician_label: label })
      });
      setFeedbackLabel(label);
    } catch (error) {
      setApiStatus(error.message);
    }
  };

  const handleEscalate = async (type) => {
    try {
      setApiStatus("");
      await requestJson(`${API_BASE}/api/clinical-outcome`, {
        method: "POST",
        headers: authHeaders(),
        body: JSON.stringify({
          patient_id: inputs.patient_id || "unknown",
          prediction_id: risk?.prediction_id,
          clinician_id: auth.username,
          outcome_label: type,
          notes: `Escalation triggered from chat triage: ${type}`,
          follow_up_days: type === "urgent_referral" ? 1 : 14
        })
      });
      setEscalationStatus(`Escalation logged: ${type}`);
      setCaseStatus("pending_review");
    } catch (error) {
      setApiStatus(error.message);
    }
  };

  const handleLoadEhr = async () => {
    try {
      if (!inputs.patient_id) return;
      setApiStatus("");
      const data = await requestJson(`${API_BASE}/api/ehr/patient/${inputs.patient_id}`, {
        headers: token ? { Authorization: `Bearer ${token}` } : {}
      });
      setEhrSummary(data);
    } catch (error) {
      setEhrSummary(null);
      setApiStatus(error.message);
    }
  };

  const handleClinicalOutcome = async () => {
    try {
      setApiStatus("");
      await requestJson(`${API_BASE}/api/clinical-outcome`, {
        method: "POST",
        headers: authHeaders(),
        body: JSON.stringify({
          patient_id: inputs.patient_id || "unknown",
          prediction_id: risk?.prediction_id,
          clinician_id: auth.username,
          outcome_label: outcomeLabel,
          notes: outcomeNotes,
          follow_up_days: 14
        })
      });
      setCaseStatus("resolved");
      setOutcomeNotes("");
    } catch (error) {
      setApiStatus(error.message);
    }
  };

  const handleTranscriptUpload = async (event) => {
    const file = event.target.files?.[0];
    if (!file) return;
    const allowed = [".txt", ".md", ".csv"];
    const ext = file.name.slice(file.name.lastIndexOf(".")).toLowerCase();
    if (!allowed.includes(ext)) {
      setApiStatus("Unsupported file format. Use .txt, .md, or .csv transcript.");
      return;
    }
    const text = await file.text();
    setChatMessage((prev) => (prev ? `${prev}\n\n${text}` : text));
  };

  const toggleCarePlanItem = (id) => {
    setCarePlanItems((prev) => prev.map((item) => (item.id === id ? { ...item, checked: !item.checked } : item)));
  };

  const toggleChatCarePlanItem = (messageId, itemId) => {
    setChatItemState((prev) => ({
      ...prev,
      [messageId]: {
        ...prev[messageId],
        checked: {
          ...(prev[messageId]?.checked || {}),
          [itemId]: !prev[messageId]?.checked?.[itemId]
        }
      }
    }));
  };

  const setChatMessageReview = (messageId, review) => {
    setChatItemState((prev) => ({
      ...prev,
      [messageId]: {
        ...prev[messageId],
        review
      }
    }));
    if (risk?.prediction_id) {
      handleFeedback(review);
    }
  };

  const openSource = async (label) => {
    try {
      const name = sourceLabelToName(label);
      const data = await requestJson(`${API_BASE}/api/rag/source/${encodeURIComponent(name)}`, {
        headers: token ? { Authorization: `Bearer ${token}` } : {}
      });
      setSourcePreview(data);
    } catch (error) {
      setApiStatus(error.message);
    }
  };

  if (!token) {
    return (
      <div className="app">
        <header className="hero">
          <div>
            <p className="eyebrow">Perinatal Preventive Decision Support</p>
            <h1>Authentication</h1>
            <p className="subtext">Login/Register to continue. Role decides landing page: Client or Clinical.</p>
          </div>
        </header>
        <section className="grid">
          <div className="card">
            <h2>Sign In</h2>
            <p className={`muted ${backendHealth.status === "offline" ? "status-bad" : "status-good"}`}>
              {backendHealth.message}
            </p>
            <div className="row">
              <input
                value={auth.username}
                onChange={(e) => setAuth((prev) => ({ ...prev, username: e.target.value }))}
                placeholder="username"
              />
              <input
                type="password"
                value={auth.password}
                onChange={(e) => setAuth((prev) => ({ ...prev, password: e.target.value }))}
                placeholder="password"
              />
              <input
                value={auth.role}
                onChange={(e) => setAuth((prev) => ({ ...prev, role: e.target.value }))}
                placeholder="patient|clinician|admin"
              />
            </div>
            <div className="row">
              <button onClick={handleRegister} disabled={authLoading}>
                {authLoading ? "Please wait..." : "Register"}
              </button>
              <button className="secondary" onClick={handleLogin} disabled={authLoading}>
                {authLoading ? "Please wait..." : "Login"}
              </button>
            </div>
            <p className="muted">API: {API_BASE}</p>
            {authStatus ? <p className="muted">{authStatus}</p> : null}
            {apiStatus ? <p className="muted">API error: {apiStatus}</p> : null}
          </div>
        </section>
      </div>
    );
  }

  return (
    <div className="app">
      <header className="hero">
        <div>
          <p className="eyebrow">Perinatal Preventive Decision Support</p>
          <h1>{role === "patient" ? "Client Portal" : "Clinical Portal"}</h1>
          <p className="subtext">Role-aware workspace for triage, risk tracking, and evidence-guided support.</p>
        </div>
        <div className="row">
          <p className="muted">Role: {role}</p>
          <button className="secondary" onClick={handleLogout}>Logout</button>
        </div>
      </header>

      <nav className="top-nav">
        {visibleNav.map((item) => (
          <button
            key={item.key}
            className={activePage === item.key ? "nav-btn active" : "nav-btn"}
            onClick={() => setActivePage(item.key)}
          >
            {item.label}
          </button>
        ))}
      </nav>

      {apiStatus ? <p className="muted">API error: {apiStatus}</p> : null}

      {activePage === "workspace" ? (
        <UnifiedWorkspace
          inputs={inputs}
          updateInput={updateInput}
          patientIdentity={patientIdentity}
          setPatientIdentity={setPatientIdentity}
          risk={risk}
          timeline={timeline}
          handleRisk={handleRisk}
          handleTimeline={handleTimeline}
          chatMessage={chatMessage}
          setChatMessage={setChatMessage}
          chatHistory={chatHistory}
          chatItemState={chatItemState}
          chatResult={chatResult}
          chatLoading={chatLoading}
          chatLocked={chatLocked}
          handleChatAssess={handleChatAssess}
          handleEscalate={handleEscalate}
          handleTranscriptUpload={handleTranscriptUpload}
          toggleChatCarePlanItem={toggleChatCarePlanItem}
          setChatMessageReview={setChatMessageReview}
          onOpenSource={openSource}
          xai={xai}
          xaiStatus={xaiStatus}
          carePlanItems={carePlanItems}
          setCarePlanItems={setCarePlanItems}
          toggleCarePlanItem={toggleCarePlanItem}
          ragResponse={ragResponse}
          chatResultSources={chatResult?.sources}
        />
      ) : null}

      {activePage === "overview" ? (
        <section className="grid">
          <div className="card">
            <h2>Quick Status</h2>
            <p className={`muted ${backendHealth.status === "offline" ? "status-bad" : "status-good"}`}>
              {backendHealth.message}
            </p>
            <p className="muted">API: {API_BASE}</p>
            {risk && typeof risk.risk_percent === "number" ? (
              <p>Latest Risk: <strong>{risk.risk_percent.toFixed(1)}%</strong></p>
            ) : (
              <p className="muted">No risk run yet.</p>
            )}
            {chatResult ? (
              <p>Latest Chat Triage: <strong>{chatResult.risk_level}</strong> ({chatResult.risk_percent}%)</p>
            ) : (
              <p className="muted">No chat triage run yet.</p>
            )}
          </div>

          <div className="card">
            <h2>Consent Setup</h2>
            <label>
              Patient ID
              <input
                value={inputs.patient_id}
                onChange={(e) => updateInput("patient_id", e.target.value)}
                placeholder="patient-001"
              />
            </label>
            <button onClick={handleConsent}>Grant Consent</button>
            <button className="secondary" onClick={handleLoadEhr}>Load EHR Summary</button>
            {consentStatus ? <p className="muted">{consentStatus}</p> : null}
            {ehrSummary ? (
              <div className="result">
                <p><strong>{ehrSummary.patient_name}</strong></p>
                <p className="muted">DOB: {ehrSummary.dob} | MRN: {ehrSummary.mrn}</p>
                <p className="muted">Recent Visits: {ehrSummary.recent_visits}</p>
                <p className="muted">Latest EPDS: {ehrSummary.latest_epds ?? "N/A"}</p>
              </div>
            ) : null}
          </div>
        </section>
      ) : null}

      {activePage === "chat" ? (
        <section className="grid">
          <div className="card">
            <h2>Chat Sentiment + RAG Risk Triage</h2>
            {chatResult ? (
              <div className="chat-risk-head">
                <span className="chip">Live Risk Meter</span>
                <RiskMeter value={chatResult.risk_percent} />
              </div>
            ) : null}
            <div className="identity-grid">
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
            <label>
              Patient Message
              <textarea
                rows="5"
                value={chatMessage}
                onChange={(e) => setChatMessage(e.target.value)}
                placeholder="Describe current emotional state, sleep, appetite, stress, and postpartum concerns."
                disabled={chatLocked}
              />
            </label>
            <button onClick={handleChatAssess} disabled={chatLoading || chatLocked}>
              {chatLoading ? "Analyzing..." : "Analyze Chat"}
            </button>
            <button
              className="secondary"
              onClick={() => {
                setChatHistory([]);
                setChatItemState({});
                setChatResult(null);
                setChatLocked(false);
                setEscalationStatus("");
              }}
            >
              Clear Chat
            </button>
            <label>
              Upload Transcript
              <input type="file" accept=".txt,.md,.csv" onChange={handleTranscriptUpload} disabled={chatLocked} />
            </label>
            {chatLocked ? (
              <div className="crisis-banner">
                Crisis mode active. Chat input locked. Use escalation actions immediately.
              </div>
            ) : null}

            {chatResult && (
              <div className={`result ${chatResult.crisis_mode ? "alert" : ""}`}>
                {chatResult.crisis_mode ? (
                  <div className="crisis-banner">Immediate Crisis Alert: escalate now</div>
                ) : null}
                <p>Risk: <strong>{chatResult.risk_percent}%</strong> ({chatResult.risk_level})</p>
                <p>Context: <strong>{chatResult.likely_context}</strong></p>
                <div className="row">
                  <span className="chip">{caseStatus.replace("_", " ")}</span>
                  {chatResult.risk_percent >= 75 || chatResult.crisis_mode ? <span className="chip chip-risk">Urgent</span> : <span className="chip chip-good">Routine</span>}
                </div>
                <div className="action-grid">
                  {chatResult.crisis_mode ? (
                    <button onClick={() => handleEscalate("emergency_services_contacted")}>
                      Escalate to Emergency
                    </button>
                  ) : null}
                  <button onClick={() => handleEscalate("urgent_referral")}>Escalate to Specialist</button>
                  <button className="secondary" onClick={() => handleEscalate("crisis_team_notified")}>Notify Crisis Team</button>
                  <a className="btn-link" href="tel:988">Call 988</a>
                </div>
                {escalationStatus ? <p className="muted">{escalationStatus}</p> : null}
                <div className="keypoint-list">
                  {toKeyPoints(chatResult.guidance).map((point, idx) => (
                    <div className="keypoint-item" key={`${point}-${idx}`}>
                      <span className="keypoint-dot" />
                      <span>{point}</span>
                    </div>
                  ))}
                </div>
                {chatResult.sources?.length ? (
                  <div className="chip-wrap">
                    {chatResult.sources.map((source) => (
                      <button key={source} className="chip chip-source" onClick={() => openSource(source)}>{source}</button>
                    ))}
                  </div>
                ) : null}
              </div>
            )}
          </div>

          <div className="card">
            <h2>Chat Thread</h2>
            {chatHistory.length ? (
              <div className="chat-thread">
                {chatHistory.map((item, index) => (
                  <ChatBubble
                    key={`${item.id || item.at}-${index}`}
                    item={item}
                    itemState={chatItemState[item.id] || { checked: {}, review: "" }}
                    onToggleCarePlan={toggleChatCarePlanItem}
                    onReview={setChatMessageReview}
                    onOpenSource={openSource}
                  />
                ))}
              </div>
            ) : (
              <p className="muted">No chat yet. Submit a message to start the thread.</p>
            )}
          </div>

          <div className="card">
            <h2>Risk Visuals</h2>
            {chatResult ? (
              <>
                <RiskMeter value={chatResult.risk_percent} />
                <p className="muted">
                  Context: <strong>{chatResult.likely_context}</strong> | Language: {chatResult.language}
                </p>
                <FactorBars factors={chatResult.risk_factors || []} />
              </>
            ) : (
              <p className="muted">Run chat analysis to see visualizations.</p>
            )}
          </div>
        </section>
      ) : null}

      {activePage === "clinical" ? (
        <section className="grid">
          <div className="card">
            <h2>Clinician Review & Feedback</h2>
            <p className="muted">Validate prediction quality and feed corrections for retraining.</p>
            <div className="row">
              <button onClick={() => handleFeedback("agree")}>Agree</button>
              <button className="secondary" onClick={() => handleFeedback(feedbackLabel)}>
                Correct
              </button>
              <button className="secondary" onClick={() => setCaseStatus("resolved")}>Sign Off</button>
              <input
                value={feedbackLabel}
                onChange={(e) => setFeedbackLabel(e.target.value)}
                placeholder="low | medium | high"
              />
            </div>
            <p className="muted">Case Status: {caseStatus.replace("_", " ")}</p>
            {risk?.prediction_id ? <p className="muted">Prediction ID: {risk.prediction_id}</p> : <p className="muted">Run risk first to submit feedback.</p>}
            <h3>Post-Consultation Outcome</h3>
            <div className="row">
              <input
                value={outcomeLabel}
                onChange={(e) => setOutcomeLabel(e.target.value)}
                placeholder="improved | stable | deteriorated"
              />
              <input
                value={outcomeNotes}
                onChange={(e) => setOutcomeNotes(e.target.value)}
                placeholder="Clinical notes"
              />
              <button className="secondary" onClick={handleClinicalOutcome}>Log Outcome</button>
            </div>
          </div>
        </section>
      ) : null}

      {activePage === "risk" ? (
        <section className="grid">
          <div className="card">
            <h2>Dynamic Risk Profiler</h2>
            <div className="form-grid">
              <label>
                Patient ID
                <input
                  value={inputs.patient_id}
                  onChange={(e) => updateInput("patient_id", e.target.value)}
                  placeholder="patient-001"
                />
              </label>
              <label>
                Gestational Weeks
                <input
                  type="number"
                  value={inputs.gestational_weeks}
                  onChange={(e) => updateInput("gestational_weeks", e.target.value)}
                />
              </label>
              <label>
                Sleep Quality (0-10)
                <input
                  type="number"
                  value={inputs.sleep_quality}
                  onChange={(e) => updateInput("sleep_quality", e.target.value)}
                />
              </label>
              <label>
                Appetite (0-10)
                <input
                  type="number"
                  value={inputs.appetite}
                  onChange={(e) => updateInput("appetite", e.target.value)}
                />
              </label>
              <label>
                Fatigue (0-10)
                <input
                  type="number"
                  value={inputs.fatigue}
                  onChange={(e) => updateInput("fatigue", e.target.value)}
                />
              </label>
              <label>
                Financial Stress (0-10)
                <input
                  type="number"
                  value={inputs.financial_stress}
                  onChange={(e) => updateInput("financial_stress", e.target.value)}
                />
              </label>
              <label>
                Self-harm Thoughts (0-5)
                <input
                  type="number"
                  value={inputs.self_harm}
                  onChange={(e) => updateInput("self_harm", e.target.value)}
                />
              </label>
              <label>
                Age
                <input
                  type="number"
                  value={inputs.age}
                  onChange={(e) => updateInput("age", e.target.value)}
                />
              </label>
              <label>
                Income Band
                <input
                  value={inputs.income_band}
                  onChange={(e) => updateInput("income_band", e.target.value)}
                  placeholder="low | middle | high"
                />
              </label>
            </div>
            <div className="row">
              <button onClick={handleRisk}>Calculate Risk</button>
              <button className="secondary" onClick={handleTimeline}>Load Timeline</button>
            </div>

            {risk && typeof risk.risk_percent === "number" ? (
              <div className={`result ${risk.crisis_mode ? "alert" : ""}`}>
                {risk.crisis_mode ? (
                  <div className="crisis-banner">Immediate Crisis Alert: circuit breaker active</div>
                ) : null}
                <p>Risk Score: <strong>{risk.risk_percent.toFixed(1)}%</strong></p>
                <p>{risk.message}</p>
              </div>
            ) : null}
          </div>

          <div className="card">
            <h2>XAI Inspector</h2>
            {xai?.contributions?.length ? (
              <XAIBarChart data={xai.contributions} />
            ) : (
              <p className="muted">
                {xaiStatus || "Run a risk score to see explainability breakdown."}
              </p>
            )}
          </div>

          <div className="card">
            <h2>Longitudinal Timeline</h2>
            {timelinePoints.length ? (
              <>
                <TimelineChart points={timelinePoints} />
                <TimelineTable points={timelinePoints} />
              </>
            ) : (
              <p className="muted">No timeline points yet.</p>
            )}
          </div>
        </section>
      ) : null}

      {activePage === "guidance" ? (
        <section className="grid">
          <div className="card">
            <h2>RAG Guidance</h2>
            <label>
              Query
              <input value={ragQuery} onChange={(e) => setRagQuery(e.target.value)} />
            </label>
            <label>
              Risk Drivers (comma-separated)
              <input value={ragDrivers} onChange={(e) => setRagDrivers(e.target.value)} />
            </label>
            <button onClick={handleRag}>Generate Guidance</button>
            {ragResponse ? (
              <div className="result">
                <div className="chip-wrap">
                  {ragDrivers
                    .split(",")
                    .map((d) => d.trim())
                    .filter(Boolean)
                    .map((driver) => (
                      <span className="chip" key={driver}>{driver}</span>
                    ))}
                </div>
                <div className="keypoint-list">
                  {toKeyPoints(ragResponse.answer).map((point, idx) => (
                    <div className="keypoint-item" key={`${point}-${idx}`}>
                      <span className="keypoint-dot" />
                      <span>{point}</span>
                    </div>
                  ))}
                </div>
                {ragResponse.sources?.length ? (
                  <div className="chip-wrap">
                    {ragResponse.sources.map((source) => (
                      <button className="chip chip-source" key={source} onClick={() => openSource(source)}>{source}</button>
                    ))}
                  </div>
                ) : null}
                <h3>Care Plan</h3>
                <div className="keypoint-list">
                  {carePlanItems.map((item) => (
                    <label key={item.id} className="row">
                      <input
                        type="checkbox"
                        checked={item.checked}
                        onChange={() => toggleCarePlanItem(item.id)}
                      />
                      <span>{item.text}</span>
                    </label>
                  ))}
                </div>
              </div>
            ) : null}
          </div>

          <div className="card">
            <h2>Fairness & Ethics</h2>
            {fairnessGroups.map((group, index) => (
              <div className="row" key={`${group.group}-${index}`}>
                <input
                  value={group.group}
                  onChange={(e) => updateGroup(index, "group", e.target.value)}
                />
                <input
                  type="number"
                  step="0.01"
                  value={group.positive_rate}
                  onChange={(e) => updateGroup(index, "positive_rate", e.target.value)}
                />
              </div>
            ))}
            <div className="row">
              <button className="secondary" onClick={addGroup}>Add Group</button>
              <button onClick={handleFairness}>Run Audit</button>
            </div>
            {fairnessResult ? (
              <div className="result">
                <div className="row">
                  <strong>Disparate Impact: {fairnessResult.disparate_impact}</strong>
                  <span
                    className={
                      fairnessResult.disparate_impact < 0.8
                        ? "chip chip-risk"
                        : "chip chip-good"
                    }
                  >
                    {fairnessResult.disparate_impact < 0.8 ? "Needs Attention" : "Acceptable"}
                  </span>
                </div>
                <RiskMeter value={Math.min(100, Math.round(fairnessResult.disparate_impact * 100))} />
                <div className="keypoint-list">
                  {toKeyPoints(fairnessResult.summary).map((point, idx) => (
                    <div className="keypoint-item" key={`${point}-${idx}`}>
                      <span className="keypoint-dot" />
                      <span>{point}</span>
                    </div>
                  ))}
                </div>
                {fairnessResult.mitigation_report?.length ? (
                  <>
                    <h3>Mitigation Report ({fairnessResult.mitigation_strategy})</h3>
                    <div className="keypoint-list">
                      {fairnessResult.mitigation_report.map((point, idx) => (
                        <div className="keypoint-item" key={`${point}-${idx}`}>
                          <span className="keypoint-dot" />
                          <span>{point}</span>
                        </div>
                      ))}
                    </div>
                  </>
                ) : null}
              </div>
            ) : null}
          </div>
        </section>
      ) : null}

      {sourcePreview ? (
        <section className="card">
          <div className="row">
            <h2>Source Preview: {sourcePreview.name}</h2>
            <button className="secondary" onClick={() => setSourcePreview(null)}>Close</button>
          </div>
          <pre className="source-preview">{sourcePreview.content}</pre>
        </section>
      ) : null}
    </div>
  );
}

function TimelineChart({ points }) {
  const width = 420;
  const height = 180;
  const padding = 24;

  const xs = points.map((p) => p.gestational_weeks);
  const ys = points.map((p) => p.risk_percent);
  const minX = Math.min(...xs, 0);
  const maxX = Math.max(...xs, 42);
  const minY = 0;
  const maxY = 100;

  const scaleX = (x) =>
    padding + ((x - minX) / (maxX - minX || 1)) * (width - padding * 2);
  const scaleY = (y) =>
    height - padding - ((y - minY) / (maxY - minY || 1)) * (height - padding * 2);

  const path = points
    .map((p, idx) => `${idx === 0 ? "M" : "L"} ${scaleX(p.gestational_weeks)} ${scaleY(p.risk_percent)}`)
    .join(" ");

  return (
    <svg className="chart" width={width} height={height}>
      <rect x="0" y="0" width={width} height={height} rx="16" className="chart-bg" />
      <line x1={padding} y1={height - padding} x2={width - padding} y2={height - padding} className="chart-axis" />
      <line x1={padding} y1={padding} x2={padding} y2={height - padding} className="chart-axis" />
      <path d={path} className="chart-line" />
      {points.map((p) => (
        <circle
          key={`${p.gestational_weeks}-${p.timestamp}`}
          cx={scaleX(p.gestational_weeks)}
          cy={scaleY(p.risk_percent)}
          r="4"
          className="chart-point"
        />
      ))}
      <text x={width - padding - 35} y={height - 8} className="chart-label">Week</text>
      <text x={8} y={padding - 4} className="chart-label">Risk %</text>
    </svg>
  );
}

function TimelineTable({ points }) {
  const rows = [...points]
    .sort((a, b) => new Date(a.timestamp) - new Date(b.timestamp))
    .slice(-6);
  return (
    <div className="timeline-table">
      {rows.map((row, idx) => (
        <div className="timeline-row" key={`${row.timestamp}-${idx}`}>
          <span>Week {row.gestational_weeks}</span>
          <strong>{Number(row.risk_percent).toFixed(1)}%</strong>
        </div>
      ))}
    </div>
  );
}

function XAIBarChart({ data }) {
  const ranked = [...data].sort((a, b) => b.contribution_percent - a.contribution_percent);
  return (
    <div className="factor-list">
      {ranked.map((item) => (
        <div className="factor-row" key={item.feature}>
          <span>{item.feature.replace(/_/g, " ")} ({item.contribution_percent}%)</span>
          <div className="factor-bar">
            <div className="factor-bar-fill xai-fill" style={{ width: `${item.contribution_percent}%` }} />
          </div>
        </div>
      ))}
    </div>
  );
}

function ChatBubble({ item, itemState, onToggleCarePlan, onReview, onOpenSource }) {
  const isUser = item.role === "user";
  const isSystem = item.role === "system";
  const isAssistant = item.role === "assistant";
  const review = itemState?.review || "";

  if (isSystem) {
    return (
      <div className="bubble system">
        <p className="bubble-role">System</p>
        <p className="bubble-text">{item.text}</p>
      </div>
    );
  }

  return (
    <div className={isUser ? "bubble user" : "bubble assistant"}>
      <p className="bubble-role">{isUser ? "Client" : "AI Assistant"}</p>
      <p className="bubble-text">
        {isUser ? highlightEvidence(item.text, item.highlightFactors || []) : item.text}
      </p>
      {item.meta?.risk_percent !== undefined ? (
        <p className="muted">Risk {item.meta.risk_percent}% ({item.meta.risk_level})</p>
      ) : null}
      {isAssistant && item.carePlan?.length ? (
        <div className="bubble-careplan">
          {item.carePlan.map((plan) => (
            <label key={plan.id} className="row bubble-check">
              <input
                type="checkbox"
                checked={Boolean(itemState?.checked?.[plan.id])}
                onChange={() => onToggleCarePlan(item.id, plan.id)}
              />
              <span>{plan.text}</span>
            </label>
          ))}
        </div>
      ) : null}
      {isAssistant ? (
        <div className="row">
          <button
            className={review === "agree" ? "" : "secondary"}
            onClick={() => onReview(item.id, "agree")}
          >
            Agree
          </button>
          <button
            className={review === "correct" ? "" : "secondary"}
            onClick={() => onReview(item.id, "correct")}
          >
            Correct
          </button>
          {item.meta?.sources?.map((source) => (
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

function RiskMeter({ value }) {
  const safeValue = Math.max(0, Math.min(100, Number(value || 0)));
  return (
    <div>
      <div className="meter-track">
        <div className="meter-fill" style={{ width: `${safeValue}%` }} />
        <div className="meter-threshold t-low" />
        <div className="meter-threshold t-mid" />
      </div>
      <div className="meter-labels">
        <span>Low</span>
        <span>Moderate</span>
        <span>High</span>
      </div>
      <p className="muted">Estimated risk: {safeValue.toFixed(1)}%</p>
    </div>
  );
}

function FactorBars({ factors }) {
  if (!factors.length) return <p className="muted">No factors detected.</p>;
  const ranked = factors.map((factor, index) => ({
    label: factor.replace(/_/g, " "),
    score: Math.max(20, 100 - index * 14)
  }));
  return (
    <div className="factor-list">
      {ranked.map((factor) => (
        <div className="factor-row" key={factor.label}>
          <span>{factor.label} ({factor.score}%)</span>
          <div className="factor-bar">
            <div className="factor-bar-fill" style={{ width: `${factor.score}%` }} />
          </div>
        </div>
      ))}
    </div>
  );
}

function toKeyPoints(text) {
  const normalized = (text || "").replace(/\s+/g, " ").trim();
  if (!normalized) return [];

  const chunks = normalized
    .split(/\.\s+|\-\s+/)
    .map((part) => part.trim())
    .filter((part) => part.length > 12);

  if (!chunks.length) return [normalized];
  return chunks.slice(0, 5);
}

function sourceLabelToName(label) {
  return (label || "").split(" (score=")[0].trim();
}

function highlightEvidence(text, factors) {
  const keywordsMap = {
    sleep_disturbance: ["sleep", "insomnia", "neend"],
    low_mood: ["sad", "hopeless", "udaas", "depressed"],
    anxiety: ["anxious", "panic", "chinta", "tension"],
    fatigue: ["fatigue", "tired", "exhausted", "thakan"],
    appetite_change: ["appetite", "bhook"],
    self_harm_ideation: ["hurt myself", "kill myself", "better off dead", "jeene ka mann nahi"]
  };
  let out = text;
  factors.forEach((factor) => {
    const impact = estimateFactorImpact(factor);
    (keywordsMap[factor] || []).forEach((word) => {
      const regex = new RegExp(`(${escapeRegExp(word)})`, "ig");
      out = out.replace(
        regex,
        `<mark class="evidence-mark" title="${factor.replace(/_/g, " ")} contributed +${impact.toFixed(
          2
        )} to risk score">$1</mark>`
      );
    });
  });
  return <span dangerouslySetInnerHTML={{ __html: out }} />;
}

function escapeRegExp(value) {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function getStoredValue(key, fallback = "") {
  try {
    const value = window.localStorage.getItem(key);
    return value ?? fallback;
  } catch {
    return fallback;
  }
}

function setStoredValue(key, value) {
  try {
    if (value === undefined || value === null || value === "") {
      window.localStorage.removeItem(key);
      return;
    }
    window.localStorage.setItem(key, String(value));
  } catch {
    // Ignore storage failures when browser blocks localStorage.
  }
}

function clearStoredSession() {
  try {
    Object.values(STORAGE_KEYS).forEach((key) => window.localStorage.removeItem(key));
  } catch {
    // Ignore storage failures when browser blocks localStorage.
  }
}

function estimateFactorImpact(factor) {
  const weights = {
    sleep_disturbance: 0.15,
    low_mood: 0.18,
    anxiety: 0.12,
    fatigue: 0.11,
    appetite_change: 0.08,
    self_harm_ideation: 0.35
  };
  return weights[factor] || 0.05;
}
