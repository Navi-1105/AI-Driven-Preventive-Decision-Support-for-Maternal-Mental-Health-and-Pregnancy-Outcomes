from pathlib import Path

import joblib

from app.config import settings
from app.models.schemas import RiskInput

_MODEL_CACHE = None


def load_model_payload():
    global _MODEL_CACHE
    if _MODEL_CACHE is not None:
        return _MODEL_CACHE

    model_path = Path(settings.model_path)
    if not model_path.is_absolute():
        model_path = Path(__file__).resolve().parents[2] / model_path

    if model_path.exists():
        _MODEL_CACHE = joblib.load(model_path)
    return _MODEL_CACHE


def map_inputs_to_features(input_data: RiskInput, features: list[str]) -> dict[str, float]:
    # Map UI inputs (0-10 scales) to dataset feature scales (0-3 where applicable)
    sleep_score = round((10.0 - input_data.behavioral.sleep_quality) / 10.0 * 3)
    appetite_score = round((10.0 - input_data.behavioral.appetite) / 10.0 * 3)
    fatigue_score = round(input_data.behavioral.fatigue / 10.0 * 3)
    sufficient_money = round((10.0 - input_data.behavioral.financial_stress) / 10.0 * 3)
    self_harm_score = round(input_data.self_harm / 5.0 * 3)

    mapped = {
        "Age": float(input_data.demographics.age),
        "Gestational Age": float(input_data.gestational_weeks),
        "Trouble falling or staying sleep or sleeping too much": float(sleep_score),
        "Poor appetite or overeating": float(appetite_score),
        "Feeling tired or having little energy": float(fatigue_score),
        "Sufficient Money for Basic Needs": float(sufficient_money),
        "Thoughts that you would be better off dead, or of hurting yourself": float(
            self_harm_score
        ),
    }

    return {feature: mapped.get(feature, 0.0) for feature in features}
