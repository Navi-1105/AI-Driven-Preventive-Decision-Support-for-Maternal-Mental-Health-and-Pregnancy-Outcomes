from app.models.schemas import FairnessRequest, FairnessResponse


def compute_disparate_impact(req: FairnessRequest) -> FairnessResponse:
    rates = [g.positive_rate for g in req.groups if g.positive_rate > 0]
    if len(rates) < 2:
        return FairnessResponse(
            protected_attribute=req.protected_attribute,
            disparate_impact=1.0,
            bias_detected=False,
            mitigation_strategy="none",
            mitigation_report=["Need at least 2 non-zero groups for bias estimation."],
            summary="Insufficient group data to compute disparate impact.",
        )

    di = min(rates) / max(rates)
    bias_detected = di < 0.8
    mitigation_strategy = "reweighting" if bias_detected else "monitor_only"
    mitigation_report = (
        [
            "Bias threshold crossed (DI < 0.8).",
            "Apply sample reweighting for under-represented group outcomes.",
            "Re-evaluate DI and equalized odds after retraining.",
        ]
        if bias_detected
        else [
            "Parity is within acceptable threshold.",
            "Continue periodic fairness monitoring.",
        ]
    )
    summary = (
        "Disparate impact closer to 1.0 indicates parity; below 0.8 may indicate bias."
    )

    return FairnessResponse(
        protected_attribute=req.protected_attribute,
        disparate_impact=round(di, 3),
        bias_detected=bias_detected,
        mitigation_strategy=mitigation_strategy,
        mitigation_report=mitigation_report,
        summary=summary,
    )
