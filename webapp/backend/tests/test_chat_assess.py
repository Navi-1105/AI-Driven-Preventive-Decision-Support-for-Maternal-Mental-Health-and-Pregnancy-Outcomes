from app.services.rag import assess_chat_message


def test_chat_assess_detects_crisis_override():
    out = assess_chat_message("After delivery I want to hurt myself.")
    assert out["crisis_mode"] is True
    assert out["risk_level"] == "high"
    assert out["risk_percent"] == 100.0
    assert "self_harm_ideation" in out["risk_factors"]


def test_chat_assess_detects_risk_factors():
    out = assess_chat_message("I feel hopeless and cannot sleep with my newborn.")
    assert "low_mood" in out["risk_factors"]
    assert "sleep_disturbance" in out["risk_factors"]
