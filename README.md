# Real-Time Social Media Analytics for E-Commerce Trends (BERT + LSTM)

Production-oriented full-stack platform with:
- `Frontend`: React + Recharts + Socket.IO real-time dashboard
- `Backend`: Flask + SQLite + JWT auth + WebSocket stream
- `NLP`: Local fine-tuned BERT sentiment model
- `Trend Intelligence`: Local trained LSTM forecasting model
- `GenAI`: Gemini API for AI summary, comparison reasoning, search insight, and dashboard insight
- `Decision Engine`: risk-based alerting and recommendation outputs
- `DevOps`: Docker, Kubernetes manifests, CI workflow, Prometheus/Grafana hooks

## Quick Start (Local)

## 1) Backend
```bash
cd backend
python -m venv .venv
. .venv/Scripts/activate  # Windows PowerShell: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
python run.py
```

Backend runs on `http://localhost:5000`.

Gemini configuration (`backend/.env`):
```bash
ENABLE_GEMINI=true
GEMINI_API_KEY=your_api_key
GEMINI_MODEL=gemini-2.0-flash
```

## Live Data + Local Training
The platform now supports live collection and local retraining:
- Live sources integrated directly via links:
  - `https://www.reddit.com/search.json`
  - `https://news.google.com/rss/search?q=<query>`
  - `https://api.twitter.com/2/tweets/search/recent` (optional token)
  - `https://www.googleapis.com/youtube/v3/search` + `commentThreads` (optional API key)
  - Google Trends via `pytrends` (optional, no API key)
- BERT and LSTM training pull live data directly from these links at training time.

## 2) Frontend
```bash
cd frontend
npm install
copy .env.example .env
npm run dev
```

Frontend runs on `http://localhost:5173`.

## Train Models (BERT + LSTM) with Live Data
From `backend/`:

```bash
.\.venv\Scripts\python.exe app\ml\train_bert_local.py
.\.venv\Scripts\python.exe app\ml\train_lstm_local.py
```

Or run end-to-end:
```bash
.\.venv\Scripts\python.exe app\ml\train_from_live.py
```

Artifacts generated:
- `backend/models/bert_sentiment/`
- `backend/models/lstm_trend.pt`

## Docker Compose
```bash
cd deploy/docker
docker compose up --build
```

Services:
- Frontend: `http://localhost:8080`
- Backend API: `http://localhost:5000`
- Prometheus: `http://localhost:9090`
- Grafana: `http://localhost:3001`

## Core API Endpoints
- `POST /api/v1/auth/register`
- `POST /api/v1/auth/login`
- `POST /api/v1/products/search`
- `GET /api/v1/products/{slug}/dashboard`
- `GET /api/v1/products/{slug}/summary`
- `POST /api/v1/products/compare`
- `GET /api/v1/products/trending`
- `GET /api/v1/models`
- `GET /api/v1/models/metrics`
- `POST /api/v1/models/register`
- `POST /api/v1/models/drift/{slug}`

Full architecture and production details: `docs/ARCHITECTURE.md`.

Filter support:
- `GET /api/v1/products/{slug}/dashboard?year=2026&month=2`
- `GET /api/v1/products/{slug}/summary?window_minutes=720&year=2026&month=2`
- `POST /api/v1/products/compare` with payload fields `year`, `month`
- `GET /api/v1/products/trending?category=smartphone&year=2026&month=2`
