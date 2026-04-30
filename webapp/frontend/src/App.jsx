import { useEffect, useMemo, useRef, useState } from "react";
import "./styles.css";
import UnifiedWorkspace from "./components/UnifiedWorkspace";
import RiskTrend from "./components/RiskTrend";
import ShapExplanation from "./components/ShapExplanation";

const API_BASE = (import.meta.env.VITE_API_BASE || "").trim();
const DEFAULT_DEV_API_BASE = "http://127.0.0.1:8000";

function resolveApiUrl(path) {
  const normalizedPath = path.startsWith("/") ? path : `/${path}`;
  // In dev, prefer Vite proxy when VITE_API_BASE is empty (proxy is configured for /api).
  if (!API_BASE && import.meta.env.DEV) return normalizedPath;
  // In prod builds with no reverse proxy, fall back to localhost backend.
  if (!API_BASE && !import.meta.env.DEV) return `${DEFAULT_DEV_API_BASE}${normalizedPath}`;
  return `${API_BASE}${normalizedPath}`;
}
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

const ROLE_OPTIONS = [
  {
    key: "patient",
    label: "Patient Portal",
    eyebrow: "Personal access",
    description: "View risk summaries, follow care guidance, and track your support plan.",
    badge: "PT"
  },
  {
    key: "clinician",
    label: "Clinician Portal",
    eyebrow: "Clinical access",
    description: "Open the unified workspace, patient review tools, and triage decisions.",
    badge: "MD"
  }
];

function calculateAgeFromDob(dob) {
  if (!dob) return null;
  const birthDate = new Date(`${dob}T00:00:00`);
  if (Number.isNaN(birthDate.getTime())) return null;

  const today = new Date();
  let age = today.getFullYear() - birthDate.getFullYear();
  const monthDiff = today.getMonth() - birthDate.getMonth();
  const dayDiff = today.getDate() - birthDate.getDate();
  if (monthDiff < 0 || (monthDiff === 0 && dayDiff < 0)) {
    age -= 1;
  }
  return age >= 0 ? age : null;
}

function normalizeTimelinePoints(data) {
  const rawPoints = Array.isArray(data) ? data : data?.points || data?.records || [];
  return rawPoints
    .map((point) => ({
      patient_id: point.patient_id,
      gestational_weeks: Number(point.gestational_weeks),
      risk_percent: Number(point.risk_percent),
      timestamp: point.timestamp
    }))
    .filter((point) => (
      Number.isFinite(point.gestational_weeks)
      && Number.isFinite(point.risk_percent)
      && point.timestamp
    ));
}

export default function App() {
  const [auth, setAuth] = useState({
    username: getStoredValue(STORAGE_KEYS.username, ""),
    password: "",
    role: "" // Start with no role selected
  });
  const [token, setToken] = useState(() => getStoredValue(STORAGE_KEYS.token, ""));
  const [currentRole, setCurrentRole] = useState(() => getStoredValue(STORAGE_KEYS.role, ""));
  const [activePage, setActivePage] = useState("overview");

  const [apiStatus, setApiStatus] = useState("");
  const [backendHealth, setBackendHealth] = useState({ status: "checking", message: "Checking backend..." });
  const [authLoading, setAuthLoading] = useState(false);
  const [authStatus, setAuthStatus] = useState("");

  const [inputs, setInputs] = useState(() => ({
    ...initialInputs,
    patient_id: getStoredValue(STORAGE_KEYS.patientId, "")
  }));
  const [patientIdentity, setPatientIdentity] = useState({
    name: "",
    dob: "",
    mrn: ""
  });
  const [patientStatus, setPatientStatus] = useState("");
  const [patientSaved, setPatientSaved] = useState(false);
  const [consentStatus, setConsentStatus] = useState("");
  const [consentGranted, setConsentGranted] = useState(false);

  const [risk, setRisk] = useState(null);
  const [xai, setXai] = useState(null);
  const [xaiStatus, setXaiStatus] = useState("");
  const [timeline, setTimeline] = useState([]);
  const [timelineStatus, setTimelineStatus] = useState("");

  const [chatMessage, setChatMessage] = useState(
    "I gave birth 6 weeks ago, I feel exhausted, cannot sleep, and feel like a bad mother."
  );
  const [chatResult, setChatResult] = useState(null);
  const [chatHistory, setChatHistory] = useState([]);
  const [chatItemState, setChatItemState] = useState({});
  const chatEndRef = useRef(null);
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
    chatEndRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [chatHistory, chatLoading]);

  useEffect(() => {
    if (!visibleNav.some((item) => item.key === activePage)) {
      setActivePage("overview");
    }
  }, [role]);

  useEffect(() => {
    let cancelled = false;

    const validateStoredSession = async () => {
      if (!token) return;
      try {
        const data = await requestJson(resolveApiUrl("/api/auth/me"), {
          headers: { Authorization: `Bearer ${token}` }
        });
        if (cancelled) return;
        setCurrentRole(data.role || "");
        setAuth((prev) => ({ ...prev, username: data.username || prev.username }));
      } catch (error) {
        if (!cancelled) {
          if (error?.status === 401) {
            clearSessionState("Session expired. Please sign in again.");
          } else {
            setApiStatus(error.message || "Could not validate saved session.");
          }
        }
      }
    };

    validateStoredSession();
    return () => {
      cancelled = true;
    };
  }, []);

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

  useEffect(() => {
    const derivedAge = calculateAgeFromDob(patientIdentity.dob);
    if (derivedAge === null) return;
    setInputs((prev) => (
      Number(prev.age) === derivedAge ? prev : { ...prev, age: derivedAge }
    ));
  }, [patientIdentity.dob]);

  const updateInput = (field, value) => {
    setInputs((prev) => ({ ...prev, [field]: value }));
    if (field === "patient_id") {
      setPatientStatus("");
      setPatientSaved(false);
      setPatientIdentity({ name: "", dob: "", mrn: "" });
      setTimeline([]);
      setTimelineStatus("");
    }
  };

  const handleRoleSelect = (nextRole) => {
    setAuth((prev) => ({ ...prev, role: nextRole }));
    setAuthStatus("");
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

  const formatApiError = (data, fallback) => {
    const detail = data?.detail;
    if (Array.isArray(detail)) {
      return detail
        .map((item) => {
          const field = Array.isArray(item.loc) ? item.loc.filter((part) => part !== "body").join(".") : "";
          return field ? `${field}: ${item.msg}` : item.msg;
        })
        .filter(Boolean)
        .join("; ");
    }
    if (typeof detail === "string") return detail;
    if (detail && typeof detail === "object") {
      return detail.msg || detail.message || JSON.stringify(detail);
    }
    return fallback;
  };

  const requestJson = async (url, options = {}, config = {}) => {
    let lastError = null;
    const timeoutMs = Number.isFinite(config.timeoutMs) ? config.timeoutMs : 15000;
    const retries = Number.isFinite(config.retries) ? Math.max(0, config.retries) : 1;
    for (let attempt = 0; attempt < retries + 1; attempt += 1) {
      const controller = new AbortController();
      const timeout = setTimeout(() => controller.abort(), timeoutMs);
      try {
        const response = await fetch(url, { ...options, signal: controller.signal });
        const data = await parseJsonSafe(response);
        if (!response.ok) {
          const message = formatApiError(data, `Request failed (${response.status})`);
          if (response.status === 401) {
            clearSessionState("Session expired. Please sign in again.");
          }
          const error = new Error(message);
          error.status = response.status;
          throw error;
        }
        return data;
      } catch (error) {
        lastError = error;
        const isRetryable = error?.name === "AbortError" || error instanceof TypeError;
        if (!isRetryable || attempt >= retries) {
          if (error?.name === "AbortError") throw new Error(`Backend timeout: ${url}`);
          throw error;
        }
      } finally {
        clearTimeout(timeout);
      }
    }
    throw lastError;
  };

  const clearSessionState = (message = "") => {
    setToken("");
    setCurrentRole("");
    setAuthStatus(message);
    setConsentStatus("");
    setConsentGranted(false);
    setPatientStatus("");
    setPatientSaved(false);
    setPatientIdentity({ name: "", dob: "", mrn: "" });
    setRisk(null);
    setXai(null);
    setTimeline([]);
    setTimelineStatus("");
    setChatResult(null);
    setChatHistory([]);
    setChatItemState({});
    setChatLocked(false);
    setRagResponse(null);
    setFairnessResult(null);
    setStoredValue(STORAGE_KEYS.token, "");
    setStoredValue(STORAGE_KEYS.role, "");
  };

  useEffect(() => {
    const checkHealth = async () => {
      const controller = new AbortController();
      const timeout = setTimeout(() => controller.abort(), 2500);
      try {
        const response = await fetch(resolveApiUrl("/api/health"), { signal: controller.signal });
        if (!response.ok) {
          throw new Error(`Health check failed (${response.status})`);
        }
        const data = await parseJsonSafe(response);
        setBackendHealth({ status: "online", message: `Backend online (${data.app || "service"})` });
      } catch {
        setBackendHealth({
          status: "offline",
          message: API_BASE
            ? `Backend offline at ${API_BASE}.`
            : import.meta.env.DEV
              ? "Backend offline. Start backend on http://127.0.0.1:8000 (Vite proxy expects /api)."
              : `Backend offline. Set VITE_API_BASE or run backend at ${DEFAULT_DEV_API_BASE}.`
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
        if (!cancelled) {
          setConsentStatus("");
          setConsentGranted(false);
        }
        return;
      }
      if (!inputs.patient_id) {
        if (!cancelled) {
          setConsentStatus("Enter patient ID to check consent");
          setConsentGranted(false);
        }
        return;
      }

      try {
        const data = await requestJson(resolveApiUrl(`/api/privacy/consent/${encodeURIComponent(inputs.patient_id)}`), {
          headers: authHeaders()
        });
        if (cancelled) return;
        if (data?.consent_given) {
          setConsentStatus(`Consent active (${(data.consent_scope || []).join(", ")})`);
          setConsentGranted(true);
        } else {
          setConsentStatus("Consent not granted");
          setConsentGranted(false);
        }
      } catch (error) {
        if (cancelled) return;
        if ((error?.message || "").toLowerCase().includes("not found")) {
          setConsentStatus("Consent not granted");
          setConsentGranted(false);
        } else {
          setConsentStatus(`Consent check failed: ${error.message}`);
          setConsentGranted(false);
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

  const fetchTimelineForPatient = async (patientId) => {
    if (!patientId) return [];
    setTimelineStatus("Loading timeline...");
    const data = await requestJson(resolveApiUrl(`/api/timeline/${encodeURIComponent(patientId)}`), {
      headers: token ? { Authorization: `Bearer ${token}` } : {}
    });
    const points = normalizeTimelinePoints(data);
    setTimeline(points);
    setTimelineStatus(
      points.length
        ? `Loaded ${points.length} timeline entr${points.length === 1 ? "y" : "ies"}.`
        : "No timeline entries yet. Run risk to create the first one."
    );
    return points;
  };

  const patientStatusClass = patientStatus.includes("found") || patientStatus.includes("saved")
    ? "status-good"
    : patientStatus.includes("New patient")
      ? "status-warn"
      : "";

  const clinicalActionsEnabled = patientSaved && consentGranted;
  const timelineActionsEnabled = Boolean(inputs.patient_id.trim()) && consentGranted;

  const fetchPatient = async (patientId = inputs.patient_id) => {
    const id = patientId.trim();
    if (!id) {
      setPatientStatus("");
      setPatientSaved(false);
      return null;
    }
    if (!token) {
      setPatientStatus("Sign in to fetch patient details.");
      return null;
    }

    try {
      setPatientStatus("Looking up patient...");
      const result = await requestJson(resolveApiUrl(`/api/patient/${encodeURIComponent(id)}`), {
        headers: token ? { Authorization: `Bearer ${token}` } : {}
      });
      if (!result.exists) {
        setPatientIdentity({ name: "", dob: "", mrn: "" });
        setPatientSaved(false);
        setTimeline([]);
        setTimelineStatus("");
        setPatientStatus("New patient");
        return null;
      }
      const patient = result.data || {};
      setPatientIdentity({
        name: patient.name || "",
        dob: patient.dob || "",
        mrn: patient.mrn || ""
      });
      setPatientSaved(true);
      setPatientStatus("Patient found");
      fetchTimelineForPatient(id).catch(() => {
        setTimelineStatus("");
      });
      return patient;
    } catch (error) {
      setPatientSaved(false);
      setPatientStatus(`Patient lookup failed: ${error.message}`);
      return null;
    }
  };

  const handlePatientBlur = () => {
    if (inputs.patient_id.trim()) {
      fetchPatient();
    }
  };

  const handleSavePatient = async () => {
    const patient_id = inputs.patient_id.trim();
    if (!patient_id) {
      setPatientStatus("Enter a Patient ID before saving.");
      return;
    }
    if (!patientIdentity.name || !patientIdentity.dob || !patientIdentity.mrn) {
      setPatientStatus("Enter name, DOB, and MRN before saving.");
      return;
    }

    try {
      setPatientStatus("Saving patient...");
      const result = await requestJson(resolveApiUrl("/api/patient"), {
        method: "POST",
        headers: authHeaders(),
        body: JSON.stringify({
          patient_id,
          name: patientIdentity.name,
          dob: patientIdentity.dob,
          mrn: patientIdentity.mrn
        })
      });
      const patient = result.data || {};
      setPatientIdentity({
        name: patient.name || "",
        dob: patient.dob || "",
        mrn: patient.mrn || ""
      });
      setPatientSaved(true);
      setPatientStatus("Patient saved");
      setTimelineStatus("Patient saved. Run risk to add timeline entries.");
    } catch (error) {
      setPatientSaved(false);
      setPatientStatus(`Patient save failed: ${error.message}`);
    }
  };

  const handleRegister = async () => {
    if (!auth.role) {
      setAuthStatus("Please select a role first");
      return;
    }
    if (!auth.username || !auth.password) {
      setAuthStatus("Please enter both username and password");
      return;
    }
    if (auth.password.length < 8) {
      setAuthStatus("Registration failed: password must be at least 8 characters.");
      return;
    }
    
    try {
      setAuthLoading(true);
      setApiStatus("");
      setAuthStatus("");
      const data = await requestJson(
        resolveApiUrl("/api/auth/register"),
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(auth)
        },
        { timeoutMs: 5000, retries: 0 }
      );
      setToken(data.access_token);
      setCurrentRole(data.role);
      setAuthStatus(`Account created. Signed in as ${data.role}.`);
      setAuth((prev) => ({ ...prev, password: "" }));
      
      // Route to role-specific page
      if (data.role === "clinician" || data.role === "admin") {
        setActivePage("workspace");
      } else if (data.role === "patient") {
        setActivePage("overview");
      } else {
        setActivePage("overview");
      }
    } catch (error) {
      setAuthStatus(`Registration failed: ${error.message || "Please try again"}`);
    } finally {
      setAuthLoading(false);
    }
  };

  const handleLogin = async () => {
    if (!auth.role) {
      setAuthStatus("Please select a role first");
      return;
    }
    if (!auth.username || !auth.password) {
      setAuthStatus("Please enter both username and password");
      return;
    }
    
    try {
      setAuthLoading(true);
      setApiStatus("");
      setAuthStatus("");
      const data = await requestJson(
        resolveApiUrl("/api/auth/login"),
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ username: auth.username, password: auth.password })
        },
        { timeoutMs: 5000, retries: 0 }
      );
      setToken(data.access_token);
      setCurrentRole(data.role);
      setAuthStatus(`Signed in as ${data.role}.`);
      setAuth((prev) => ({ ...prev, password: "" }));
      
      // Route to role-specific page
      if (data.role === "clinician" || data.role === "admin") {
        setActivePage("workspace");
      } else if (data.role === "patient") {
        setActivePage("overview");
      } else {
        setActivePage("overview");
      }
    } catch (error) {
      setAuthStatus(`Sign-in failed: ${error.message || "Invalid credentials"}`);
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
    setConsentGranted(false);
    setPatientStatus("");
    setPatientSaved(false);
    setPatientIdentity({ name: "", dob: "", mrn: "" });
    setRisk(null);
    setXai(null);
    setTimeline([]);
    setChatResult(null);
    setChatHistory([]);
    setChatItemState({});
    setChatLocked(false);
    setRagResponse(null);
    setFairnessResult(null);
    // Reset auth form
    setAuth({
      username: "",
      password: "",
      role: ""
    });
    clearStoredSession();
  };

  const handleConsent = async () => {
    try {
      if (!inputs.patient_id) {
        setConsentStatus("Enter a Patient ID before granting consent.");
        return;
      }
      if (!token) {
        setConsentStatus("Sign in before granting consent.");
        return;
      }
      setApiStatus("");
      setConsentStatus("Saving consent...");
      const data = await requestJson(resolveApiUrl("/api/privacy/consent"), {
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
        const consentActive = Boolean(data.consent_given);
        setConsentGranted(consentActive);
        setConsentStatus(
          consentActive
            ? `Consent active (${(data.consent_scope || []).join(", ")})`
            : "Consent not granted"
        );
        if (consentActive) {
          fetchTimelineForPatient(inputs.patient_id).catch((error) => {
            setTimelineStatus(getClinicalErrorMessage(error));
          });
        }
      }
    } catch (error) {
      setConsentStatus(error.message || "Consent failed");
      setConsentGranted(false);
      setApiStatus(error.message || "Consent failed");
    }
  };

  const getClinicalErrorMessage = (error) => {
    const message = error?.message || "Request failed";
    if (message === "Consent record required for patient") {
      return "Consent record required for patient. Enter the Patient ID and click Grant Consent first.";
    }
    if (message === "Consent not granted by patient") {
      return "Consent not granted by patient. Click Grant Consent before running clinical tools.";
    }
    if (message.startsWith("Consent scope missing:")) {
      return `${message}. Update consent before running this workflow.`;
    }
    return message;
  };

  const handleRisk = async () => {
    try {
      setApiStatus("");
      const payload = buildPayload();
      const data = await requestJson(resolveApiUrl("/api/risk"), {
        method: "POST",
        headers: authHeaders(),
        body: JSON.stringify(payload)
      });
      setRisk(data);

      const xaiData = await requestJson(resolveApiUrl("/api/xai"), {
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
        await fetchTimelineForPatient(inputs.patient_id);
      }
    } catch (error) {
      setRisk(null);
      setXai(null);
      setXaiStatus("");
      setApiStatus(getClinicalErrorMessage(error));
    }
  };

  const handleTimeline = async () => {
    try {
      const patientId = inputs.patient_id.trim();
      if (!patientId) {
        setTimelineStatus("Enter a Patient ID before loading timeline.");
        return;
      }
      if (!consentGranted) {
        setTimelineStatus("Grant consent before loading timeline.");
        return;
      }
      setApiStatus("");
      if (!patientSaved) {
        const patient = await fetchPatient(patientId);
        if (!patient) {
          setTimelineStatus("Save this patient before loading timeline.");
          return;
        }
      }
      await fetchTimelineForPatient(patientId);
    } catch (error) {
      setTimeline([]);
      const message = getClinicalErrorMessage(error);
      setTimelineStatus(message);
      setApiStatus(message);
    }
  };

  const handleChatAssess = async () => {
    if (chatLocked) return;
    const message = chatMessage.trim();
    if (!message) {
      setApiStatus("Type a patient message before sending.");
      return;
    }
    setChatLoading(true);
    try {
      setApiStatus("");
      const runId = `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
      const userMessage = {
        id: `u-${runId}`,
        role: "user",
        text: message,
        at: new Date().toISOString()
      };
      const data = await requestJson(resolveApiUrl("/api/chat-assess"), {
        method: "POST",
        headers: authHeaders(),
        body: JSON.stringify({ patient_id: inputs.patient_id || undefined, message })
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
      // Initialize care plan items from chat guidance
      const newCarePlanItems = carePlan.map((item) => ({ 
        ...item, 
        checked: false,
        id: item.id || `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`
      }));
      setCarePlanItems(newCarePlanItems);
      setChatMessage("");
    } catch (error) {
      setChatResult(null);
      setApiStatus(getClinicalErrorMessage(error));
    } finally {
      setChatLoading(false);
    }
  };

  const handleRag = async () => {
    try {
      setApiStatus("");
      const data = await requestJson(resolveApiUrl("/api/rag"), {
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
      const data = await requestJson(resolveApiUrl("/api/fairness"), {
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
      await requestJson(resolveApiUrl("/api/feedback"), {
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
      await requestJson(resolveApiUrl("/api/clinical-outcome"), {
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
      if (!["clinician", "admin"].includes(role)) {
        setApiStatus("EHR summary is available only in the clinician portal.");
        return;
      }
      if (!inputs.patient_id) {
        setApiStatus("Enter a Patient ID before loading EHR summary.");
        return;
      }
      setApiStatus("");
      const data = await requestJson(resolveApiUrl(`/api/patient/${encodeURIComponent(inputs.patient_id)}`), {
        headers: token ? { Authorization: `Bearer ${token}` } : {}
      });
      if (!data.exists || !data.data) {
        setEhrSummary(null);
        setPatientSaved(false);
        setPatientStatus("New patient");
        return;
      }
      const patient = data.data;
      setPatientIdentity({
        name: patient.name || "",
        dob: patient.dob || "",
        mrn: patient.mrn || ""
      });
      setPatientSaved(true);
      setPatientStatus("Patient found");
      setEhrSummary({
        patient_name: patient.name,
        dob: patient.dob,
        mrn: patient.mrn,
        recent_visits: 0,
        latest_epds: null
      });
    } catch (error) {
      setEhrSummary(null);
      setApiStatus(error.message);
    }
  };

  const handleClinicalOutcome = async () => {
    try {
      setApiStatus("");
      await requestJson(resolveApiUrl("/api/clinical-outcome"), {
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
    
    const allowed = [".txt", ".md", ".csv", ".pdf"];
    const ext = file.name.slice(file.name.lastIndexOf(".")).toLowerCase();
    
    if (!allowed.includes(ext)) {
      setApiStatus("Unsupported file format. Supported formats: .txt, .md, .csv, or .pdf");
      return;
    }

    try {
      setApiStatus("");
      let text = "";

      if (ext === ".pdf") {
        // Extract text from PDF
        text = await extractTextFromPDF(file);
      } else {
        // Read text files directly
        text = await file.text();
      }

      if (!text || text.trim().length === 0) {
        setApiStatus("File appears to be empty or could not be read.");
        return;
      }

      // Add file info header
      const fileInfo = `[Uploaded: ${file.name} - ${(file.size / 1024).toFixed(1)} KB]\n\n`;
      setChatMessage((prev) => (prev ? `${prev}\n\n${fileInfo}${text}` : `${fileInfo}${text}`));
      setApiStatus(
        `Loaded ${file.name} (${(file.size / 1024).toFixed(1)} KB, ${text.split("\n").length} lines)`
      );
      
      // Clear the file input so the same file can be uploaded again if needed
      event.target.value = "";
    } catch (error) {
      setApiStatus(`Error: ${error.message || "Failed to process file"}`);
      console.error("File upload error:", error);
    }
  };

  // PDF text extraction function using PDF.js
  const extractTextFromPDF = async (file) => {
    try {
      // Check if PDF.js is loaded (from index.html)
      const pdfjsLib = window.pdfjsLib || window.pdfjs;
      
      if (!pdfjsLib) {
        throw new Error("PDF.js library not loaded. Please refresh the page.");
      }

      // Set worker source if not already set
      if (!pdfjsLib.GlobalWorkerOptions.workerSrc) {
        pdfjsLib.GlobalWorkerOptions.workerSrc = "https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js";
      }

      setApiStatus("Extracting text from PDF...");
      
      const arrayBuffer = await file.arrayBuffer();
      const loadingTask = pdfjsLib.getDocument({ 
        data: arrayBuffer,
        useSystemFonts: true
      });
      const pdf = await loadingTask.promise;
      
      let fullText = "";
      const totalPages = pdf.numPages;
      
      // Extract text from all pages
      for (let pageNum = 1; pageNum <= totalPages; pageNum++) {
        const page = await pdf.getPage(pageNum);
        const textContent = await page.getTextContent();
        const pageText = textContent.items
          .map((item) => item.str)
          .join(" ")
          .trim();
        
        if (pageText) {
          fullText += `${pageText}\n\n`;
        }
      }
      
      if (!fullText.trim()) {
        throw new Error("No text content found in PDF. The PDF may contain only images (scanned document).");
      }
      
      return fullText.trim();
    } catch (error) {
      console.error("PDF extraction error:", error);
      throw new Error(
        `PDF extraction failed: ${error.message}. ` +
        `Note: Scanned PDFs (image-based) are not supported. ` +
        `Please use a PDF with selectable text or convert to .txt format.`
      );
    }
  };

  const toggleCarePlanItem = (id) => {
    setCarePlanItems((prev) => {
      const updated = prev.map((item) => {
        const itemId = item.id || item;
        const matchId = typeof id === "string" ? id : (id?.id || id);
        if (itemId === matchId || item === id) {
          return { ...item, checked: !(item.checked || false) };
        }
        return item;
      });
      return updated;
    });
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
      const data = await requestJson(resolveApiUrl(`/api/rag/source/${encodeURIComponent(name)}`), {
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
            <h1>Secure Access</h1>
            <p className="subtext">Select a portal to continue.</p>
          </div>
        </header>
        <section className="auth-section">
          <div className="auth-card">
            <h2>Sign in</h2>
            <p className={`muted ${backendHealth.status === "offline" ? "status-bad" : "status-good"}`}>
              {backendHealth.message}
            </p>

            <div className="role-selection">
              {ROLE_OPTIONS.map((option) => (
                <button
                  key={option.key}
                  type="button"
                  className={`role-btn ${auth.role === option.key ? "selected" : ""}`}
                  onClick={() => handleRoleSelect(option.key)}
                >
                  <div className="role-icon">{option.badge}</div>
                  <div className="role-info">
                    <h3>{option.label}</h3>
                    <p className="muted">{option.description}</p>
                  </div>
                </button>
              ))}
            </div>

            <div className="form-group">
              <label>
                Username
                <input
                  value={auth.username}
                  onChange={(e) => setAuth((prev) => ({ ...prev, username: e.target.value }))}
                  placeholder={auth.role === "patient" ? "patient-001" : "clin1"}
                  autoFocus
                />
              </label>

              <label>
                Password
                <input
                  type="password"
                  value={auth.password}
                  onChange={(e) => setAuth((prev) => ({ ...prev, password: e.target.value }))}
                  placeholder="Enter your password"
                  onKeyDown={(e) => {
                    if (e.key === "Enter" && auth.username && auth.password && auth.role) {
                      handleLogin();
                    }
                  }}
                />
                <span className="field-hint">Use at least 8 characters for sign up.</span>
              </label>
            </div>

            <div className="auth-actions">
              <button
                onClick={handleLogin}
                disabled={
                  authLoading ||
                  backendHealth.status === "offline" ||
                  !auth.username ||
                  !auth.password ||
                  !auth.role
                }
                className="primary-btn"
              >
                {authLoading ? "Signing in..." : "Sign In"}
              </button>
              <button
                onClick={handleRegister}
                disabled={
                  authLoading ||
                  backendHealth.status === "offline" ||
                  !auth.username ||
                  !auth.password ||
                  !auth.role
                }
                className="secondary-btn"
              >
                {authLoading ? "Creating..." : "Sign Up"}
              </button>
            </div>

            {authStatus ? (
              <div
                className={`auth-status ${
                  authStatus.toLowerCase().includes("failed") || authStatus.toLowerCase().includes("error")
                    ? "error"
                    : "success"
                }`}
              >
                {authStatus}
              </div>
            ) : null}

            {apiStatus ? <div className="auth-status error">API error: {apiStatus}</div> : null}

            <p className="muted auth-footer">
              API Endpoint:{" "}
              {API_BASE
                ? API_BASE
                : import.meta.env.DEV
                  ? "(via Vite proxy: /api → http://127.0.0.1:8000)"
                  : `(${DEFAULT_DEV_API_BASE} fallback — set VITE_API_BASE to override)`}
            </p>
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
          patientStatus={patientStatus}
          patientStatusClass={patientStatusClass}
          clinicalActionsEnabled={clinicalActionsEnabled}
          timelineActionsEnabled={timelineActionsEnabled}
          timelineStatus={timelineStatus}
          risk={risk}
          timeline={timeline}
          handleRisk={handleRisk}
          handleTimeline={handleTimeline}
          handlePatientBlur={handlePatientBlur}
          handleSavePatient={handleSavePatient}
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
            <h2>{role === "patient" ? "Patient Consent" : "Consent & EHR"}</h2>
            <label>
              Patient ID
              <input
                value={inputs.patient_id}
                onChange={(e) => updateInput("patient_id", e.target.value)}
                onBlur={handlePatientBlur}
                placeholder="patient-001"
              />
            </label>
            <button onClick={handleConsent}>Grant Consent</button>
            {["clinician", "admin"].includes(role) ? (
              <button className="secondary" onClick={handleLoadEhr}>Load EHR Summary</button>
            ) : null}
            {consentStatus ? <p className="muted">{consentStatus}</p> : null}
            {patientStatus ? <p className={`muted ${patientStatusClass}`}>{patientStatus}</p> : null}
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
        <section className="chatgpt-page">
          <aside className="chatgpt-sidebar">
            <div className="chatgpt-panel">
              <h2>Patient Context</h2>
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
                {patientStatus ? <span className={`muted ${patientStatusClass}`}>{patientStatus}</span> : null}
              </div>
            </div>

            <div className="chatgpt-panel">
              <h2>Live Triage</h2>
              {chatResult ? (
                <>
                  <RiskMeter value={chatResult.risk_percent} />
                  <div className="chatgpt-status-list">
                    <div><span>Risk</span><strong>{chatResult.risk_level}</strong></div>
                    <div><span>Context</span><strong>{chatResult.likely_context}</strong></div>
                    <div><span>Case</span><strong>{caseStatus.replace("_", " ")}</strong></div>
                  </div>
                  <FactorBars factors={chatResult.risk_factors || []} />
                </>
              ) : (
                <p className="muted">Conversation insights appear after the first message.</p>
              )}
            </div>
          </aside>

          <main className="chatgpt-main">
            <div className="chatgpt-header">
              <div>
                <p className="eyebrow">Chat Triage</p>
                <h2>Maternal Mental Health Assistant</h2>
              </div>
              <div className="row">
                {chatResult ? (
                  <span className={`chip ${chatResult.risk_percent >= 75 || chatResult.crisis_mode ? "chip-risk" : "chip-good"}`}>
                    {chatResult.risk_percent}% {chatResult.risk_level}
                  </span>
                ) : null}
                <button
                  className="secondary"
                  onClick={() => {
                    setChatHistory([]);
                    setChatItemState({});
                    setChatResult(null);
                    setChatLocked(false);
                    setEscalationStatus("");
                    setChatMessage("");
                  }}
                >
                  New Chat
                </button>
              </div>
            </div>

            <div className="chatgpt-thread" aria-live="polite">
              {!chatHistory.length ? (
                <div className="chatgpt-empty">
                  <h3>Start a patient conversation</h3>
                  <div className="chatgpt-prompts">
                    {[
                      "I have not been sleeping and feel overwhelmed.",
                      "I feel anxious most of the day and cannot stop crying.",
                      "I feel exhausted and have lost interest in eating."
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
              ) : (
                chatHistory.map((item, index) => (
                  <ChatBubble
                    key={`${item.id || item.at}-${index}`}
                    item={item}
                    itemState={chatItemState[item.id] || { checked: {}, review: "" }}
                    onToggleCarePlan={toggleChatCarePlanItem}
                    onReview={setChatMessageReview}
                    onOpenSource={openSource}
                  />
                ))
              )}
              {chatLoading ? (
                <div className="bubble assistant typing">
                  <p className="bubble-role">Assistant</p>
                  <div className="typing-dots"><span /><span /><span /></div>
                </div>
              ) : null}
              <div ref={chatEndRef} />
            </div>

            {chatLocked ? (
              <div className="crisis-banner">
                Crisis mode active. Chat input locked. Use escalation actions immediately.
              </div>
            ) : null}

            {chatResult?.crisis_mode || chatResult?.risk_percent >= 75 ? (
              <div className="chatgpt-escalation-bar">
                <button onClick={() => handleEscalate("urgent_referral")}>Escalate to Specialist</button>
                <button className="secondary" onClick={() => handleEscalate("crisis_team_notified")}>Notify Crisis Team</button>
                <a className="btn-link" href="tel:988">Call 988</a>
                {escalationStatus ? <span className="muted">{escalationStatus}</span> : null}
              </div>
            ) : null}

            <div className="chatgpt-composer">
              <textarea
                rows="3"
                value={chatMessage}
                onChange={(e) => setChatMessage(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !e.shiftKey) {
                    e.preventDefault();
                    handleChatAssess();
                  }
                }}
                placeholder="Message the triage assistant..."
                disabled={chatLocked || chatLoading}
              />
              <div className="composer-actions">
                <label className="upload-pill">
                  Upload
                  <input
                    className="sr-only"
                    type="file"
                    accept=".txt,.md,.csv,.pdf"
                    onChange={handleTranscriptUpload}
                    disabled={chatLocked}
                  />
                </label>
                <button onClick={handleChatAssess} disabled={chatLoading || chatLocked || !chatMessage.trim() || !clinicalActionsEnabled}>
                  {chatLoading ? "Analyzing..." : "Send"}
                </button>
              </div>
            </div>
          </main>
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
        <section className="risk-dashboard-page">
          <div className="card risk-input-card">
            <h2>Dynamic Risk Profiler</h2>
            <div className="form-grid">
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
                  readOnly
                  title="Age is calculated from patient DOB"
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
              <button onClick={handleRisk} disabled={!clinicalActionsEnabled}>Calculate Risk</button>
              <button className="secondary" onClick={handleConsent}>Grant Consent</button>
              <button className="secondary" onClick={handleTimeline} disabled={!timelineActionsEnabled}>Load Timeline</button>
            </div>
            {consentStatus ? <p className="muted">{consentStatus}</p> : null}
            {patientStatus ? <p className={`muted ${patientStatusClass}`}>{patientStatus}</p> : null}
            {timelineStatus ? <p className="muted">{timelineStatus}</p> : null}
            {timelinePoints.length ? (
              <div>
                <h3>Timeline History</h3>
                <TimelineTable points={timelinePoints} />
              </div>
            ) : null}

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

          <div className="card risk-overview-card">
            <h2>Risk Output Dashboard</h2>
            <RiskOutputDashboard
              risk={risk}
              inputs={inputs}
              timelinePoints={timelinePoints}
            />
          </div>

          <div className="card risk-distribution-card">
            <h2>Risk Distribution / Trend Graph</h2>
            <RiskDistributionTrendGraph
              timelinePoints={timelinePoints}
              currentRisk={risk?.risk_percent}
              gestationalWeeks={Number(inputs.gestational_weeks)}
            />
            <div className="risk-trend-panel">
              <RiskTrend
                timelinePoints={timelinePoints}
                currentRisk={risk?.risk_percent}
                gestationalWeeks={Number(inputs.gestational_weeks)}
              />
            </div>
          </div>

          <div className="card risk-explainability-card">
            <h2>SHAP Explainability</h2>
            {xai?.contributions?.length || risk?.crisis_mode ? (
              <ShapExplanation
                contributions={xai?.contributions || []}
                riskPercent={risk?.risk_percent}
                crisisMode={risk?.crisis_mode}
              />
            ) : (
              <p className="muted">
                {xaiStatus || "Run a risk score to see explainability breakdown."}
              </p>
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
    .sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp));
  return (
    <div className="timeline-table">
      {rows.map((row, idx) => (
        <div className="timeline-row" key={`${row.timestamp}-${idx}`}>
          <span>
            Week {row.gestational_weeks}
            <small>{formatTimelineDate(row.timestamp)}</small>
          </span>
          <strong>{Number(row.risk_percent).toFixed(1)}%</strong>
        </div>
      ))}
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

function RiskOutputDashboard({ risk, inputs, timelinePoints }) {
  const currentRisk = typeof risk?.risk_percent === "number" ? Number(risk.risk_percent) : null;
  const currentLevel = describeRiskLevel(currentRisk);
  const currentWeek = Number(inputs.gestational_weeks || 0);
  const priorPoints = [...(timelinePoints || [])]
    .sort((a, b) => new Date(a.timestamp) - new Date(b.timestamp));
  const latestHistorical = priorPoints.length ? priorPoints[priorPoints.length - 1] : null;
  const delta = currentRisk !== null && latestHistorical ? currentRisk - Number(latestHistorical.risk_percent) : null;
  const trajectory = currentRisk === null || delta === null
    ? "Awaiting longitudinal comparison"
    : delta > 3
      ? "Escalating"
      : delta < -3
        ? "Improving"
        : "Stable";

  return (
    <div className="risk-output-dashboard">
      <div className="risk-dashboard-stats">
        <DashboardStat
          label="Current risk"
          value={currentRisk !== null ? `${currentRisk.toFixed(1)}%` : "Pending"}
          tone={currentLevel.tone}
        />
        <DashboardStat
          label="Risk band"
          value={currentLevel.label}
          tone={currentLevel.tone}
        />
        <DashboardStat
          label="Gestational week"
          value={currentWeek ? `Week ${currentWeek}` : "Not set"}
          tone="neutral"
        />
        <DashboardStat
          label="Trajectory"
          value={trajectory}
          detail={delta !== null ? `${delta > 0 ? "+" : ""}${delta.toFixed(1)} pts vs prior` : "Load or create timeline data"}
          tone={trajectory === "Escalating" ? "high" : trajectory === "Improving" ? "low" : "neutral"}
        />
      </div>

      <div className="risk-dashboard-meter">
        <div className="risk-meter-heading">
          <span className="chip">Risk severity</span>
          <span className={`chip ${currentLevel.chipClass}`}>{currentLevel.label}</span>
        </div>
        <RiskMeter value={currentRisk ?? 0} />
        <p className="muted">
          {risk?.message || "Run the risk calculation to populate the dashboard and narrative output."}
        </p>
      </div>
    </div>
  );
}

function DashboardStat({ label, value, detail, tone = "neutral" }) {
  return (
    <div className={`dashboard-stat ${tone}`}>
      <span className="dashboard-stat-label">{label}</span>
      <strong className="dashboard-stat-value">{value}</strong>
      {detail ? <span className="dashboard-stat-detail">{detail}</span> : null}
    </div>
  );
}

function RiskDistributionTrendGraph({ timelinePoints = [], currentRisk = null, gestationalWeeks = null }) {
  const combined = [...timelinePoints];
  if (typeof currentRisk === "number" && gestationalWeeks !== null && !Number.isNaN(gestationalWeeks)) {
    combined.push({
      gestational_weeks: gestationalWeeks,
      risk_percent: currentRisk,
      timestamp: new Date().toISOString(),
      isCurrent: true
    });
  }

  if (!combined.length) {
    return <p className="muted">Run a risk score and load timeline data to populate the distribution graph.</p>;
  }

  const buckets = [
    { label: "Low", min: 0, max: 40, tone: "low" },
    { label: "Moderate", min: 40, max: 70, tone: "moderate" },
    { label: "High", min: 70, max: 101, tone: "high" }
  ].map((bucket) => ({
    ...bucket,
    count: combined.filter((point) => point.risk_percent >= bucket.min && point.risk_percent < bucket.max).length
  }));

  const maxCount = Math.max(...buckets.map((bucket) => bucket.count), 1);
  const sortedTimeline = [...combined]
    .sort((a, b) => new Date(a.timestamp) - new Date(b.timestamp))
    .slice(-6);

  return (
    <div className="risk-distribution-graph">
      <div className="distribution-bars">
        {buckets.map((bucket) => (
          <div key={bucket.label} className="distribution-bar-group">
            <div className="distribution-bar-track">
              <div
                className={`distribution-bar-fill ${bucket.tone}`}
                style={{ height: `${(bucket.count / maxCount) * 100}%` }}
              />
            </div>
            <strong>{bucket.count}</strong>
            <span>{bucket.label}</span>
          </div>
        ))}
      </div>

      <div className="distribution-summary">
        <p className="muted">
          Distribution across {combined.length} recorded risk event{combined.length === 1 ? "" : "s"}.
        </p>
        <div className="timeline-mini-table">
          {sortedTimeline.map((point, index) => (
            <div
              key={`${point.timestamp}-${index}`}
              className={`timeline-mini-row ${point.isCurrent ? "current" : ""}`}
            >
              <span>Week {point.gestational_weeks}</span>
              <strong>{Number(point.risk_percent).toFixed(1)}%</strong>
            </div>
          ))}
        </div>
      </div>
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
      <p className="bubble-role">{isUser ? "Client" : "Assistant"}</p>
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

function describeRiskLevel(value) {
  const safeValue = typeof value === "number" ? Math.max(0, Math.min(100, value)) : null;
  if (safeValue === null) {
    return { label: "Pending", tone: "neutral", chipClass: "chip-source" };
  }
  if (safeValue > 70) {
    return { label: "High risk", tone: "high", chipClass: "chip-risk" };
  }
  if (safeValue > 40) {
    return { label: "Moderate risk", tone: "moderate", chipClass: "" };
  }
  return { label: "Low risk", tone: "low", chipClass: "chip-good" };
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
