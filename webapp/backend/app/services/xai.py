from typing import Dict, List

import numpy as np
import pandas as pd

from app.models.schemas import RiskInput, XAIContribution
from app.utils.model import load_model_payload, map_inputs_to_features


def normalize_contributions(contributions: Dict[str, float]) -> List[XAIContribution]:
    total = sum(max(v, 0.0) for v in contributions.values())
    if total <= 0:
        return [XAIContribution(feature=k, contribution_percent=0.0) for k in contributions]

    return [
        XAIContribution(
            feature=feature,
            contribution_percent=round((max(value, 0.0) / total) * 100.0, 2),
        )
        for feature, value in contributions.items()
    ]


def explain_with_shap(input_data: RiskInput) -> List[XAIContribution] | None:
    payload = load_model_payload()
    if payload is None:
        return None

    try:
        import shap  # type: ignore
    except Exception:
        return None

    features = payload["features"]
    pipeline = payload["pipeline"]
    classes = payload.get("label_classes", [])

    mapped = map_inputs_to_features(input_data, features)
    X = pd.DataFrame([[mapped[f] for f in features]], columns=features)

    model = pipeline.named_steps.get("rf", pipeline)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    if isinstance(shap_values, list):
        if classes and "Depressed" in classes:
            class_index = classes.index("Depressed")
        else:
            class_index = 1 if len(shap_values) > 1 else 0
        values = shap_values[class_index][0]
    else:
        arr = np.asarray(shap_values)
        if classes and "Depressed" in classes:
            class_index = classes.index("Depressed")
        else:
            class_index = 1

        # Handle SHAP output variants:
        # (n_samples, n_features, n_classes), (n_samples, n_features), or (n_features,)
        if arr.ndim == 3:
            safe_class_index = min(class_index, arr.shape[2] - 1)
            values = arr[0, :, safe_class_index]
        elif arr.ndim == 2:
            values = arr[0]
        elif arr.ndim == 1:
            values = arr
        else:
            return None

    contributions = {feature: float(abs(values[i])) for i, feature in enumerate(features)}
    total = sum(contributions.values())
    if total <= 0:
        return [XAIContribution(feature=f, contribution_percent=0.0) for f in features]

    return [
        XAIContribution(feature=f, contribution_percent=round((v / total) * 100.0, 2))
        for f, v in contributions.items()
    ]
