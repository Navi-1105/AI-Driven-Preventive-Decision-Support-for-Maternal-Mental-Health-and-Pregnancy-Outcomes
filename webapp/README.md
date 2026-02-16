# Perinatal Preventive Decision Support (Webapp)

This folder contains a FastAPI + MongoDB backend and a React frontend scaffold for the research paper demo.

## Backend

### Setup

```bash
cd webapp/backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Run

```bash
uvicorn app.main:app --reload --port 8000
```

Environment variables (optional):

- `MONGO_URI` (default: `mongodb://localhost:27017`)
- `MONGO_DB` (default: `perinatal_ai`)
- `MODEL_PATH` (default: `model/risk_model.joblib`)
- `CHAT_MODEL_PATH` (default: `model/chat_risk_model.joblib`)
- `ALERT_WEBHOOK_URL` (optional clinician alert webhook endpoint)
- `CLINICIAN_CONTACT` (default: `on-call-clinician`)
- `EMERGENCY_CONTACT` (default: `988`)
- `ENABLE_MULTILINGUAL_CHAT` (default: `true`)

## Frontend

### Setup

```bash
cd webapp/frontend
npm install
```

### Run

```bash
npm run dev
```

The frontend expects the API at `http://localhost:8000`.

## Model Training

Train the Random Forest model from the oversampled dataset:

```bash
cd webapp/backend
python scripts/train_model.py --csv ../oversampled_data.csv --out model/risk_model.joblib
```

This will save the model used by `/api/risk` and `/api/xai`.
Metrics are written to `model/metrics/risk_metrics.json`.

Train chat sentiment/risk model (TF-IDF + Logistic Regression):

```bash
cd webapp/backend
python scripts/train_chat_model.py --out model/chat_risk_model.joblib
```

Optional: provide your own labeled chat CSV (`text,label`):

```bash
python scripts/train_chat_model.py --csv /path/to/chat_dataset.csv --out model/chat_risk_model.joblib
```
Metrics are written to `model/metrics/chat_model_metrics.json`.

Retrain risk model from clinician feedback (versioned output):

```bash
python scripts/retrain_model_from_feedback.py --csv ../oversampled_data.csv --model-dir model --mongo-uri mongodb://localhost:27017 --mongo-db perinatal_ai --min-feedback 50
```

This creates:

- `model/risk_model_v<timestamp>.joblib`
- `model/risk_model.joblib` (latest pointer copy)
- `model/metrics/risk_retrain_<timestamp>.json`

Build vector RAG index:

```bash
python scripts/build_rag_index.py
```

## SHAP Explanations

The XAI inspector will use SHAP values when the model and `shap` dependency are available.
If SHAP is not installed, it falls back to normalized feature-importance heuristics.

## Chat Sentiment + RAG Triage

Use `POST /api/chat-assess` with:

- `patient_id` (optional)
- `message` (free-text user chat)

The API returns:

- `risk_percent`, `risk_level`, and likely `postpartum/antenatal` context
- detected `language` and `model_used` (`rules_only` or hybrid ML)
- detected `risk_factors` and `protective_factors`
- `crisis_mode` if self-harm language appears
- `alert_status` for crisis dispatch (`sent_webhook`, `queued_local`, `webhook_failed`)
- RAG-grounded guidance and `sources`

## Auth + Privacy Workflow

All sensitive clinical endpoints now require bearer auth and patient consent.

1. Register/login

```bash
curl -X POST http://localhost:8000/api/auth/register -H "Content-Type: application/json" -d '{"username":"clin1","password":"StrongPass123","role":"clinician"}'
curl -X POST http://localhost:8000/api/auth/login -H "Content-Type: application/json" -d '{"username":"clin1","password":"StrongPass123"}'
```

2. Upsert consent

```bash
curl -X POST http://localhost:8000/api/privacy/consent -H "Authorization: Bearer <TOKEN>" -H "Content-Type: application/json" -d '{"patient_id":"patient-001","consent_given":true,"consent_scope":["chat_assessment","risk_scoring","model_improvement"]}'
```

3. Call protected endpoints (`/api/risk`, `/api/xai`, `/api/chat-assess`, `/api/timeline/...`) with bearer token.

Role controls:

- `patient|clinician|admin`: risk, xai, timeline, rag, chat-assess
- `clinician|admin`: fairness endpoints, feedback
- `admin`: retrain request endpoint

## Fairness Automation

Run auto fairness from stored prediction logs:

```bash
curl -X POST http://localhost:8000/api/fairness/auto -H "Authorization: Bearer <TOKEN>" -H "Content-Type: application/json" -d '{"protected_attribute":"income_band","positive_threshold":60}'
```

## Tests and CI

Backend unit tests:

```bash
cd webapp/backend
pytest -q
```

CI workflow:

- `.github/workflows/webapp-ci.yml`
