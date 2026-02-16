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
    classes = payload["label_classes"]

    mapped = map_inputs_to_features(input_data, features)
    X = pd.DataFrame([[mapped[f] for f in features]], columns=features)
    proba = pipeline.predict_proba(X)[0]

    try:
        depressed_idx = classes.index("Depressed")
    except ValueError:
        depressed_idx = int(np.argmax(proba))

    risk_percent = float(proba[depressed_idx] * 100.0)

    # Use feature importances as a proxy for contributions (placeholder for SHAP)
    rf = pipeline.named_steps.get("rf")
    importances = getattr(rf, "feature_importances_", np.ones(len(features)))
    contribution_raw = {
        feature: float(importances[i] * abs(mapped[feature]))
        for i, feature in enumerate(features)
    }
    return risk_percent, contribution_raw, "Random Forest prediction computed."


def compute_risk(input_data: RiskInput) -> Tuple[RiskScore, Dict[str, float]]:
    # Hard-coded crisis override
    if input_data.self_harm > 0:
        return (
            RiskScore(
                risk_percent=100.0,
                crisis_mode=True,
                message=(
                    "Crisis mode activated: immediate support is required. "
                    "Please contact local emergency services and a clinician." 
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
