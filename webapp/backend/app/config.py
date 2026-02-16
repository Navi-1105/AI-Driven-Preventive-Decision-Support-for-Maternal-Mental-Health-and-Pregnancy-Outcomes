from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "Perinatal Preventive Decision Support"
    mongo_uri: str = "mongodb://localhost:27017"
    mongo_db: str = "perinatal_ai"
    model_path: str = "model/risk_model.joblib"
    chat_model_path: str = "model/chat_risk_model.joblib"
    alert_webhook_url: str = ""
    clinician_contact: str = "on-call-clinician"
    emergency_contact: str = "988"
    enable_multilingual_chat: bool = True


settings = Settings()
