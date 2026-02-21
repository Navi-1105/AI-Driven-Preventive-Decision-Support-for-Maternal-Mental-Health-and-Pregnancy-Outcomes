from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


class BehavioralInputs(BaseModel):
    sleep_quality: float = Field(..., ge=0, le=10)
    appetite: float = Field(..., ge=0, le=10)
    fatigue: float = Field(..., ge=0, le=10)
    financial_stress: float = Field(..., ge=0, le=10)


class Demographics(BaseModel):
    age: int = Field(..., ge=12, le=55)
    income_band: str = Field(..., description="e.g., low, middle, high")


class RiskInput(BaseModel):
    patient_id: Optional[str] = None
    gestational_weeks: int = Field(..., ge=1, le=42)
    behavioral: BehavioralInputs
    demographics: Demographics
    self_harm: int = Field(0, ge=0, le=5)
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class RiskScore(BaseModel):
    prediction_id: Optional[str] = None
    risk_percent: float
    crisis_mode: bool
    message: str


class XAIContribution(BaseModel):
    feature: str
    contribution_percent: float


class XAIResponse(BaseModel):
    risk_percent: float
    contributions: List[XAIContribution]


class TimelinePoint(BaseModel):
    gestational_weeks: int
    risk_percent: float
    timestamp: datetime


class TimelineResponse(BaseModel):
    patient_id: str
    points: List[TimelinePoint]


class RAGRequest(BaseModel):
    query: str
    risk_drivers: List[str] = []


class RAGResponse(BaseModel):
    answer: str
    sources: List[str]


class ChatAssessRequest(BaseModel):
    patient_id: Optional[str] = None
    message: str = Field(..., min_length=3, max_length=5000)


class ChatAssessResponse(BaseModel):
    risk_percent: float
    risk_level: str
    likely_context: str
    language: str
    model_used: str
    risk_factors: List[str]
    protective_factors: List[str]
    crisis_mode: bool
    alert_status: Optional[str] = None
    guidance: str
    sources: List[str]
    note: str


class FairnessGroupMetric(BaseModel):
    group: str
    positive_rate: float


class FairnessRequest(BaseModel):
    protected_attribute: str
    groups: List[FairnessGroupMetric]


class FairnessResponse(BaseModel):
    protected_attribute: str
    disparate_impact: float
    bias_detected: bool
    mitigation_strategy: str
    mitigation_report: List[str] = []
    summary: str


class FeedbackRequest(BaseModel):
    prediction_id: str
    clinician_id: Optional[str] = None
    clinician_label: str = Field(..., description="e.g., low, medium, high")
    notes: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class UserRegisterRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=40)
    password: str = Field(..., min_length=8, max_length=128)
    role: str = Field(default="patient", pattern="^(patient|clinician|admin)$")


class UserLoginRequest(BaseModel):
    username: str
    password: str


class AuthTokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    role: str
    username: str


class ConsentUpsertRequest(BaseModel):
    patient_id: str
    consent_given: bool
    consent_scope: List[str] = ["chat_assessment", "risk_scoring", "model_improvement"]
    updated_by: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ConsentResponse(BaseModel):
    patient_id: str
    consent_given: bool
    consent_scope: List[str]
    timestamp: datetime


class FairnessAutoRequest(BaseModel):
    protected_attribute: str = "income_band"
    positive_threshold: float = 60.0
    eval_thresholds: List[float] = Field(default_factory=lambda: [40.0, 50.0, 60.0, 70.0])


class RetrainRequest(BaseModel):
    min_feedback_samples: int = 50


class RetrainResponse(BaseModel):
    status: str
    message: str
    model_version: Optional[str] = None


class ClinicalOutcomeRequest(BaseModel):
    patient_id: str
    prediction_id: Optional[str] = None
    clinician_id: Optional[str] = None
    outcome_label: str = Field(..., description="e.g., improved, stable, deteriorated")
    notes: Optional[str] = None
    follow_up_days: int = 14
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class EHRPatientSummary(BaseModel):
    patient_id: str
    patient_name: str
    dob: str
    mrn: str
    latest_epds: Optional[int] = None
    recent_visits: int = 0
    known_conditions: List[str] = []
