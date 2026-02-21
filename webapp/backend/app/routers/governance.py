from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException

from app.config import settings
from app.db.mongo import get_database
from app.deps.auth import require_roles
from app.models.schemas import (
    ConsentResponse,
    ConsentUpsertRequest,
    FairnessAutoRequest,
    FairnessRequest,
    FairnessResponse,
    RetrainRequest,
    RetrainResponse,
)
from app.services.fairness import compute_disparate_impact

router = APIRouter(prefix="/api", tags=["governance"])


@router.post("/privacy/consent", response_model=ConsentResponse)
async def upsert_consent(
    payload: ConsentUpsertRequest,
    user=Depends(require_roles("patient", "clinician", "admin")),
):
    db = get_database()
    data = payload.model_dump()
    if user["role"] == "patient" and payload.updated_by not in (None, user["username"]):
        raise HTTPException(status_code=403, detail="Patients can only update their own consent")

    await db.consents.update_one(
        {"patient_id": payload.patient_id},
        {"$set": data},
        upsert=True,
    )

    await db.audit_logs.insert_one(
        {
            "event": "consent_updated",
            "patient_id": payload.patient_id,
            "updated_by": payload.updated_by or user["username"],
            "consent_given": payload.consent_given,
            "scope": payload.consent_scope,
        }
    )

    return ConsentResponse(
        patient_id=payload.patient_id,
        consent_given=payload.consent_given,
        consent_scope=payload.consent_scope,
        timestamp=payload.timestamp,
    )


@router.get("/privacy/consent/{patient_id}", response_model=ConsentResponse)
async def get_consent(
    patient_id: str,
    user=Depends(require_roles("patient", "clinician", "admin")),
):
    db = get_database()
    item = await db.consents.find_one({"patient_id": patient_id}, {"_id": 0})
    if item is None:
        raise HTTPException(status_code=404, detail="Consent record not found")

    return ConsentResponse(**item)


@router.post("/fairness/auto", response_model=FairnessResponse)
async def fairness_auto(
    payload: FairnessAutoRequest,
    user=Depends(require_roles("clinician", "admin")),
):
    db = get_database()
    cursor = db.predictions.find({}, {"demographics": 1, "risk_percent": 1, "_id": 0})
    groups: dict[str, dict[str, list[float] | int]] = {}
    valid_thresholds = sorted(
        {float(t) for t in ([payload.positive_threshold] + payload.eval_thresholds) if 0 <= float(t) <= 100}
    )
    if not valid_thresholds:
        raise HTTPException(status_code=400, detail="At least one threshold in [0, 100] is required")

    async for row in cursor:
        demographics = row.get("demographics", {})
        group = str(demographics.get(payload.protected_attribute, "unknown"))
        risk = float(row.get("risk_percent", 0.0))

        if group not in groups:
            groups[group] = {"total": 0, "risks": []}
        groups[group]["total"] += 1
        groups[group]["risks"].append(risk)

    fairness_groups = []
    for group, values in groups.items():
        total = int(values["total"])
        if total == 0:
            continue
        risks = values["risks"]
        operating_positives = sum(1 for score in risks if score >= payload.positive_threshold)
        fairness_groups.append(
            {
                "group": group,
                "positive_rate": operating_positives / total,
            }
        )

    if not fairness_groups:
        raise HTTPException(status_code=400, detail="No prediction data available for auto fairness")

    result = compute_disparate_impact(
        FairnessRequest(
            protected_attribute=payload.protected_attribute,
            groups=fairness_groups,
        )
    )

    # Sensitivity analysis across multiple decision thresholds.
    di_by_threshold = {}
    for threshold in valid_thresholds:
        rates = []
        for values in groups.values():
            total = int(values["total"])
            if total == 0:
                continue
            risks = values["risks"]
            rates.append(sum(1 for score in risks if score >= threshold) / total)
        non_zero_rates = [rate for rate in rates if rate > 0]
        if len(non_zero_rates) >= 2:
            di_by_threshold[str(int(threshold) if threshold.is_integer() else threshold)] = round(
                min(non_zero_rates) / max(non_zero_rates), 3
            )

    if di_by_threshold:
        worst_di = min(di_by_threshold.values())
        best_di = max(di_by_threshold.values())
        result.mitigation_report.extend(
            [
                (
                    f"Operating threshold={payload.positive_threshold:.1f}% "
                    "is used for triage alignment and workload control."
                ),
                (
                    "Threshold sensitivity (disparate impact by threshold): "
                    + ", ".join([f"{k}%->{v}" for k, v in di_by_threshold.items()])
                ),
                (
                    f"Robustness range across thresholds: min DI={worst_di}, max DI={best_di}. "
                    "Report this range with the operating-point DI."
                ),
                (
                    "Threshold-free AUC-gap requires ground-truth outcome labels per group; "
                    "prediction logs alone are insufficient."
                ),
            ]
        )

    await db.audit_logs.insert_one(
        {
            "event": "fairness_auto_run",
            "protected_attribute": payload.protected_attribute,
            "positive_threshold": payload.positive_threshold,
            "eval_thresholds": valid_thresholds,
            "di_by_threshold": di_by_threshold,
            "result": result.model_dump(),
            "run_by": user["username"],
        }
    )
    return result


@router.post("/ml/retrain", response_model=RetrainResponse)
async def request_retrain(
    payload: RetrainRequest,
    user=Depends(require_roles("admin")),
):
    db = get_database()
    version = Path(settings.model_path).stem
    await db.retrain_requests.insert_one(
        {
            "requested_by": user["username"],
            "min_feedback_samples": payload.min_feedback_samples,
            "current_model": version,
            "status": "queued",
        }
    )

    return RetrainResponse(
        status="queued",
        message=(
            "Retrain request queued. Run scripts/retrain_model_from_feedback.py in backend to generate a new versioned model."
        ),
        model_version=None,
    )
