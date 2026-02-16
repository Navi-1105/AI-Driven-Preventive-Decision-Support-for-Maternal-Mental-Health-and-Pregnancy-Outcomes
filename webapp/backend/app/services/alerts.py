import json
from datetime import datetime, timezone
from urllib import request

from app.config import settings


def dispatch_crisis_alert(patient_id: str, message: str, risk_percent: float) -> str:
    payload = {
        "event": "crisis_mode",
        "patient_id": patient_id,
        "risk_percent": risk_percent,
        "clinician_contact": settings.clinician_contact,
        "emergency_contact": settings.emergency_contact,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "message_excerpt": message[:300],
    }

    if not settings.alert_webhook_url:
        return "queued_local"

    try:
        req = request.Request(
            settings.alert_webhook_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with request.urlopen(req, timeout=5):
            pass
        return "sent_webhook"
    except Exception:
        return "webhook_failed"
