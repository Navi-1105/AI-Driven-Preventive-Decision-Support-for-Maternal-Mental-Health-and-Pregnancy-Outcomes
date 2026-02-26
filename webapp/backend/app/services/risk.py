from typing import Dict, Tuple

import numpy as np
import pandas as pd

from app.models.schemas import RiskInput, RiskScore
from app.utils.model import load_model_payload, map_inputs_to_features


def _model_predict(input_data: RiskInput) -> Tuple[float, Dict[str, float], str]:
    payload = load_model_payload()
    if payload is None:
        raise FileNotFoundError("Model not found")

    features = payload["features"]
    pipeline = payload["pipeline"]
    classes = payload.get("label_classes", [])

    mapped = map_inputs_to_features(input_data, features)
    X = pd.DataFrame([[mapped[f] for f in features]], columns=features)
    proba = pipeline.predict_proba(X)[0]

    depressed_idx = 1 if len(proba) > 1 else int(np.argmax(proba))
    if classes:
        cls_lower = [str(c).strip().lower() for c in classes]
        if "depressed" in cls_lower:
            depressed_idx = cls_lower.index("depressed")
        elif "1" in cls_lower:
            depressed_idx = cls_lower.index("1")

    risk_percent = float(proba[depressed_idx] * 100.0)

    estimator = None
    for step_name in ("xgb", "rf"):
        if step_name in pipeline.named_steps:
            estimator = pipeline.named_steps[step_name]
            break
    if estimator is None:
        estimator = pipeline.steps[-1][1]

    importances = getattr(estimator, "feature_importances_", np.ones(len(features)))
    if len(importances) != len(features):
        importances = np.ones(len(features))
    contribution_raw = {
        feature: float(importances[i] * abs(mapped[feature]))
        for i, feature in enumerate(features)
    }
    model_name = estimator.__class__.__name__
    return risk_percent, contribution_raw, f"{model_name} prediction computed."


def _crisis_override(input_data: RiskInput) -> tuple[bool, str]:
    if input_data.self_harm > 0:
        return True, "self-harm indicator"
    if (
        input_data.behavioral.sleep_quality <= 2
        and input_data.behavioral.appetite <= 2
        and input_data.behavioral.fatigue >= 9
    ):
        return True, "critical behavioral threshold cluster"
    return False, ""


def compute_risk(input_data: RiskInput) -> Tuple[RiskScore, Dict[str, float]]:
    crisis_mode, reason = _crisis_override(input_data)
    if crisis_mode:
        return (
            RiskScore(
                risk_percent=100.0,
                crisis_mode=True,
                message=(
                    "Crisis mode activated: immediate support is required. "
                    f"Trigger: {reason}. Please contact local emergency services and a clinician."
                ),
            ),
            {"self_harm": 100.0},
        )

    try:
        risk_percent, contributions, message = _model_predict(input_data)
        return (
            RiskScore(risk_percent=risk_percent, crisis_mode=False, message=message),
            contributions,
        )
    except Exception:
        # Simple weighted heuristic (fallback when model is unavailable)
        weights = {
            "gestational_weeks": 0.25,
            "sleep_quality": 0.25,
            "appetite": 0.15,
            "fatigue": 0.2,
            "financial_stress": 0.15,
        }

        gestation_norm = (input_data.gestational_weeks / 42.0) * 10.0

        raw = (
            gestation_norm * weights["gestational_weeks"]
            + (10.0 - input_data.behavioral.sleep_quality) * weights["sleep_quality"]
            + (10.0 - input_data.behavioral.appetite) * weights["appetite"]
            + input_data.behavioral.fatigue * weights["fatigue"]
            + input_data.behavioral.financial_stress * weights["financial_stress"]
        )

        risk_percent = max(0.0, min(100.0, (raw / 10.0) * 100.0))
        message = "Heuristic risk score (model not loaded)."

        contributions = {
            "gestational_weeks": gestation_norm * weights["gestational_weeks"],
            "sleep_quality": (10.0 - input_data.behavioral.sleep_quality)
            * weights["sleep_quality"],
            "appetite": (10.0 - input_data.behavioral.appetite) * weights["appetite"],
            "fatigue": input_data.behavioral.fatigue * weights["fatigue"],
            "financial_stress": input_data.behavioral.financial_stress
            * weights["financial_stress"],
        }

        return (
            RiskScore(risk_percent=risk_percent, crisis_mode=False, message=message),
            contributions,
        )
