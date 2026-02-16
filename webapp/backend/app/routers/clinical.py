from fastapi import APIRouter, Depends, HTTPException

from app.db.mongo import get_database
from app.deps.auth import require_roles
from app.models.schemas import (
    ChatAssessRequest,
    ChatAssessResponse,
    ClinicalOutcomeRequest,
    EHRPatientSummary,
    FairnessRequest,
    FairnessResponse,
    FeedbackRequest,
    RAGRequest,
    RAGResponse,
    RiskInput,
    RiskScore,
    TimelineResponse,
    XAIResponse,
)
from app.services.fairness import compute_disparate_impact
from app.services.alerts import dispatch_crisis_alert
from app.services.rag import (
    assess_chat_message,
    generate_answer,
    generate_chat_guidance,
    get_source_content,
)
from app.services.risk import compute_risk
from app.services.xai import explain_with_shap, normalize_contributions

router = APIRouter(prefix="/api")


async def _enforce_consent(patient_id: str | None, scope: str):
    if not patient_id:
        return
    db = get_database()
    consent = await db.consents.find_one({"patient_id": patient_id}, {"_id": 0})
    if consent is None:
        raise HTTPException(status_code=403, detail="Consent record required for patient")
    if not consent.get("consent_given", False):
        raise HTTPException(status_code=403, detail="Consent not granted by patient")
    scope_list = consent.get("consent_scope", [])
    if scope not in scope_list:
        raise HTTPException(status_code=403, detail=f"Consent scope missing: {scope}")


@router.post("/risk", response_model=RiskScore)
async def calculate_risk(
    payload: RiskInput, user=Depends(require_roles("patient", "clinician", "admin"))
):
    await _enforce_consent(payload.patient_id, "risk_scoring")
    db = get_database()
    score, contributions = compute_risk(payload)

    record = {**payload.model_dump(), **score.model_dump(), "contributions": contributions}
    result = await db.predictions.insert_one(record)
    prediction_id = str(result.inserted_id)
    await db.predictions.update_one(
        {"_id": result.inserted_id}, {"$set": {"prediction_id": prediction_id}}
    )
    score.prediction_id = prediction_id

    # store timeline point
    await db.timeline.insert_one(
        {
            "patient_id": payload.patient_id or "unknown",
            "gestational_weeks": payload.gestational_weeks,
            "risk_percent": score.risk_percent,
            "timestamp": payload.timestamp,
        }
    )

    return score


@router.post("/xai", response_model=XAIResponse)
async def explain_risk(
    payload: RiskInput, user=Depends(require_roles("patient", "clinician", "admin"))
):
    await _enforce_consent(payload.patient_id, "risk_scoring")
    score, contributions = compute_risk(payload)
    if score.crisis_mode:
        return XAIResponse(risk_percent=score.risk_percent, contributions=[])

    shap_contributions = explain_with_shap(payload)
    if shap_contributions:
        return XAIResponse(risk_percent=score.risk_percent, contributions=shap_contributions)

    normalized = normalize_contributions(contributions)
    return XAIResponse(risk_percent=score.risk_percent, contributions=normalized)


@router.get("/timeline/{patient_id}", response_model=TimelineResponse)
async def get_timeline(
    patient_id: str, user=Depends(require_roles("patient", "clinician", "admin"))
):
    await _enforce_consent(patient_id, "risk_scoring")
    db = get_database()
    cursor = db.timeline.find({"patient_id": patient_id}).sort("timestamp", 1)
    points = [
        {
            "gestational_weeks": item["gestational_weeks"],
            "risk_percent": item["risk_percent"],
            "timestamp": item["timestamp"],
        }
        async for item in cursor
    ]

    return TimelineResponse(patient_id=patient_id, points=points)


@router.post("/rag", response_model=RAGResponse)
async def rag_assistant(
    payload: RAGRequest, user=Depends(require_roles("patient", "clinician", "admin"))
):
    answer, sources = generate_answer(payload.query, payload.risk_drivers)
    return RAGResponse(answer=answer, sources=sources)


@router.get("/rag/source/{source_name}")
async def rag_source_content(
    source_name: str, user=Depends(require_roles("patient", "clinician", "admin"))
):
    source = get_source_content(source_name)
    if source is None:
        raise HTTPException(status_code=404, detail="Source not found")
    return source


@router.post("/chat-assess", response_model=ChatAssessResponse)
async def chat_assess(
    payload: ChatAssessRequest, user=Depends(require_roles("patient", "clinician", "admin"))
):
    await _enforce_consent(payload.patient_id, "chat_assessment")
    db = get_database()
    assessment = assess_chat_message(payload.message)
    guidance, sources = generate_chat_guidance(
        payload.message,
        assessment["risk_factors"],
        assessment["likely_context"],
        assessment["crisis_mode"],
    )

    note = (
        "Screening support only. This is not a diagnosis and should be reviewed by a clinician."
    )

    patient_id = payload.patient_id or "unknown"
    alert_status = None
    if assessment["crisis_mode"]:
        alert_status = dispatch_crisis_alert(
            patient_id=patient_id,
            message=payload.message,
            risk_percent=assessment["risk_percent"],
        )

    response = ChatAssessResponse(
        risk_percent=assessment["risk_percent"],
        risk_level=assessment["risk_level"],
        likely_context=assessment["likely_context"],
        language=assessment["language"],
        model_used=assessment["model_used"],
        risk_factors=assessment["risk_factors"],
        protective_factors=assessment["protective_factors"],
        crisis_mode=assessment["crisis_mode"],
        alert_status=alert_status,
        guidance=guidance,
        sources=sources,
        note=note,
    )

    await db.chat_assessments.insert_one({"patient_id": patient_id, "message": payload.message, **response.model_dump()})
    if alert_status is not None:
        await db.alerts.insert_one(
            {
                "patient_id": patient_id,
                "status": alert_status,
                "risk_percent": response.risk_percent,
                "message_excerpt": payload.message[:300],
            }
        )
    return response


@router.post("/fairness", response_model=FairnessResponse)
async def fairness_audit(
    payload: FairnessRequest, user=Depends(require_roles("clinician", "admin"))
):
    return compute_disparate_impact(payload)


@router.post("/feedback")
async def clinician_feedback(
    payload: FeedbackRequest, user=Depends(require_roles("clinician", "admin"))
):
    db = get_database()
    prediction = await db.predictions.find_one({"prediction_id": payload.prediction_id})
    if prediction is None:
        raise HTTPException(status_code=404, detail="Prediction not found")

    await db.feedback.insert_one(payload.model_dump())
    return {"status": "saved"}


@router.get("/ehr/patient/{patient_id}", response_model=EHRPatientSummary)
async def ehr_patient_summary(
    patient_id: str, user=Depends(require_roles("clinician", "admin"))
):
    db = get_database()
    item = await db.ehr_mock.find_one({"patient_id": patient_id}, {"_id": 0})
    if item:
        return EHRPatientSummary(**item)

    # Mock fallback to keep EHR integration point ready
    return EHRPatientSummary(
        patient_id=patient_id,
        patient_name="Unknown Patient",
        dob="N/A",
        mrn=f"MRN-{patient_id}",
        latest_epds=None,
        recent_visits=0,
        known_conditions=[],
    )


@router.post("/clinical-outcome")
async def clinical_outcome(
    payload: ClinicalOutcomeRequest, user=Depends(require_roles("clinician", "admin"))
):
    db = get_database()
    await db.clinical_outcomes.insert_one(payload.model_dump())
    return {"status": "saved"}
