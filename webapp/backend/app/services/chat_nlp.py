from pathlib import Path
from typing import Dict, List, Tuple

import joblib

from app.config import settings

_CHAT_MODEL_CACHE = None


def load_chat_model_payload():
    global _CHAT_MODEL_CACHE
    if _CHAT_MODEL_CACHE is not None:
        return _CHAT_MODEL_CACHE

    model_path = Path(settings.chat_model_path)
    if not model_path.is_absolute():
        model_path = Path(__file__).resolve().parents[2] / model_path

    if not model_path.exists():
        return None

    _CHAT_MODEL_CACHE = joblib.load(model_path)
    return _CHAT_MODEL_CACHE


def predict_chat_risk(message: str) -> Tuple[float, str, List[str], str] | None:
    payload = load_chat_model_payload()
    if payload is None:
        return None

    pipeline = payload["pipeline"]
    label_classes = payload.get("label_classes", ["low", "moderate", "high"])
    factor_map: Dict[str, List[str]] = payload.get("factor_map", {})

    probabilities = pipeline.predict_proba([message])[0]

    # Priority target for this project: estimate postpartum depression risk probability
    if "high" in label_classes:
        high_index = label_classes.index("high")
    else:
        high_index = int(probabilities.argmax())

    risk_percent = float(round(probabilities[high_index] * 100.0, 1))

    if risk_percent >= 75:
        risk_level = "high"
    elif risk_percent >= 45:
        risk_level = "moderate"
    else:
        risk_level = "low"

    text = message.lower()
    factors: List[str] = []
    for factor, terms in factor_map.items():
        if any(term in text for term in terms):
            factors.append(factor)

    return risk_percent, risk_level, factors, "tfidf_logreg"
