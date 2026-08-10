# Enterprise Architecture Blueprint

## 1. Text-Based System Architecture Diagram

```text
                               +-----------------------+
                               |   Web / Mobile UI     |
                               |  React + Socket.IO    |
                               +-----------+-----------+
                                           |
                                           v
                              +------------+-------------+
                              | API Gateway / BFF Layer  |
                              | Flask REST + JWT + RBAC  |
                              +-----+------+-----+-------+
                                    |      |     |
                    +---------------+      |     +------------------+
                    |                      |                        |
                    v                      v                        v
         +----------+-----------+  +-------+---------+    +--------+---------+
         | Ingestion Service    |  | NLP Service     |    | Decision Engine  |
         | Reddit/X/News APIs   |  | BERT Sentiment  |    | Risk Scoring     |
         | Rate-aware collectors|  | Summary models  |    | Alerts + actions |
         +----------+-----------+  +-------+---------+    +--------+---------+
                    |                      |                        |
                    +-----------+----------+------------------------+
                                |
                                v
                    +-----------+------------------+
                    | Trend Engine (LSTM-ready)    |
                    | Temporal aggregation/forecast |
                    +-----------+------------------+
                                |
              +-----------------+------------------+
              |                                    |
              v                                    v
   +----------+-------------+         +------------+------------+
   | SQLite (OLTP metadata) |         | Cache/Stream (optional) |
   | users/posts/metrics    |         | Redis/Kafka (cloud mode)|
   +----------+-------------+         +------------+------------+
              |                                    |
              v                                    v
   +----------+-------------+         +------------+------------+
   | MLOps Registry         |         | Observability Stack     |
   | model versions/drift   |         | Prometheus/Grafana/logs |
   +------------------------+         +-------------------------+
```

## 2. Technology Stack and Justification

- `Frontend`: React + Recharts + Socket.IO client.
Reason: highly interactive dashboard, lightweight component model, good chart ecosystem.
- `Backend`: Flask + Flask-SocketIO + JWT.
Reason: fast iteration, modular API blueprints, easy WebSocket integration.
- `Database`: SQLite (current requirement), microservice-ready DAL.
Reason: zero-admin local persistence, easy portability to PostgreSQL in production.
- `NLP`: Hugging Face Transformers (BERT sentiment), fallback lightweight scoring.
Reason: contextual sentiment beats lexicon-only systems on noisy social text.
- `Temporal Forecasting`: LSTM-ready service interface with retrain hooks.
Reason: captures sequence dynamics from sentiment and mention windows.
- `MLOps`: model registry table + drift monitoring endpoint.
Reason: lifecycle traceability for governance and retraining.
- `Observability`: Prometheus metrics + Grafana dashboards + structured logs.
Reason: production SLO visibility and incident triage.
- `Deployment`: Docker + Kubernetes + GitHub Actions CI.
Reason: reproducibility, portability, and cloud-native release model.

## 3. Database Schema (SQLite)

Core entities:
- `users`: authentication and role mapping.
- `products`: canonical product catalog from user search.
- `search_sessions`: user-product query sessions.
- `social_posts`: ingested multi-source posts with sentiment/risk tags.
- `product_metrics`: hourly aggregates (mentions, sentiment, engagement, trend score).
- `trend_forecasts`: future mentions/sentiment predictions.
- `risk_events`: risk engine outputs (severity, trigger, status).
- `model_registry`: model lifecycle metadata (version, metrics, artifact URI).

Key relationships:
- `products 1:N social_posts`
- `products 1:N product_metrics`
- `products 1:N trend_forecasts`
- `products 1:N risk_events`
- `users 1:N search_sessions`

## 4. API Design (REST + Realtime)

Authentication:
- `POST /api/v1/auth/register`
- `POST /api/v1/auth/login`

Product analytics:
- `POST /api/v1/products/search`: trigger live ingestion + analytics.
- `GET /api/v1/products/{slug}/dashboard`: timeline, sentiment distribution, risks.
- `GET /api/v1/products/{slug}/summary`: AI positive/negative/overall summary.
- `GET /api/v1/products/trending?limit=10`: top trend-score products.
- `POST /api/v1/products/compare`: head-to-head comparison + recommendation.
- `GET /api/v1/products/{slug}/risks`: risk event stream.

Model lifecycle:
- `GET /api/v1/models`
- `POST /api/v1/models/register`
- `POST /api/v1/models/drift/{slug}`

Realtime:
- WebSocket namespace: `/stream`
- Events: `connected`, `subscribe_product`, `analytics_update`

## 5. ML Model Selection and Training Strategy

Sentiment model:
- Base: `distilbert-base-uncased-finetuned-sst-2-english` (default).
- Fine-tuning strategy:
  - domain data: e-commerce and social slang corpora.
  - class balancing: focal loss or weighted sampling.
  - validation: macro-F1 + calibration error.

Trend model:
- Input sequence features per time bucket:
  - avg sentiment
  - mention count
  - engagement sum
  - derived momentum/acceleration
- LSTM strategy:
  - sliding windows (e.g., 24-hour sequence)
  - train/val split by time to avoid leakage
  - metrics: MAE, RMSE, directional accuracy

Lifecycle:
- Register model versions in `model_registry`.
- Track drift using sentiment distribution shift checks.
- Retraining policy:
  - scheduled monthly + event-triggered retrain if drift score threshold exceeded.

## 6. Security Architecture

Identity and access:
- JWT authentication with role claims.
- Route-level protection with `@jwt_required`.

Data security:
- Password hashing (`werkzeug` hash API).
- Input sanitization and normalization pipeline.
- Secrets externalized via env vars and K8s secrets.

API hardening:
- CORS policy control.
- Request validation and explicit error boundaries.
- Rate-aware collectors to avoid upstream abuse and account lockouts.

Operational security:
- Separate dev/prod secrets.
- Container image scanning in CI/CD (recommended extension).
- Audit trails via risk and model event tables.

## 7. Privacy, Compliance, and Governance

Privacy:
- Store only necessary fields from public posts.
- Optional pseudonymization/tokenization for author IDs.
- Retention policy: TTL-based cleanup job for old raw text.

Compliance controls:
- Data lineage per source and timestamp.
- Explainable risk decisions (trigger + details).
- Model registry captures version and evaluation metrics.

Governance:
- Policy-driven model promotion (`shadow -> canary -> active`).
- Drift monitoring endpoint for operational risk.
- Human-in-the-loop override for high-severity risk actions.

## 8. Deployment Architecture (Docker + Kubernetes)

Docker:
- `backend/Dockerfile`: Flask API + Socket.IO + Gunicorn/eventlet.
- `frontend/Dockerfile`: Vite build + Nginx static serving.
- `deploy/docker/docker-compose.yml`: backend, frontend, Prometheus, Grafana.

Kubernetes:
- `deploy/k8s/backend.yaml`: backend deployment + SQLite PVC.
- `deploy/k8s/frontend.yaml`: horizontally scalable frontend.
- `deploy/k8s/ingress.yaml`: ingress routing for UI and `/api`.
- `deploy/k8s/hpa.yaml`: autoscaling policy (frontend baseline).

Cloud-ready note:
- SQLite supports local/single-writer deployments.
- For true multi-replica backend, migrate to PostgreSQL + Redis/Kafka.

## 9. Scalability Strategy

Current:
- Modular service boundaries in Flask package.
- Async realtime push via Socket.IO.
- Aggregated metrics reduce dashboard query load.

Scale-up roadmap:
- Split services: API gateway, ingestion workers, NLP inference service, trend service.
- Introduce queue/stream backbone (Kafka/Flink) for burst ingestion.
- Replace SQLite with PostgreSQL + TimescaleDB for write concurrency.
- Cache hot results and trend snapshots in Redis.

## 10. Failure Handling Strategy

Upstream API failures:
- Connector-level try/catch with source fallback.
- Mock data mode for resilience in demos and testing.

Pipeline failures:
- Partial commit strategy: failed post skipped, batch continues.
- Error boundaries on API routes (400/401/404/500 consistent responses).

Model failures:
- Transformer unavailable -> fallback sentiment scorer.
- Drift warnings trigger retrain workflow.

Operational failures:
- Health endpoint and Prometheus scraping.
- Container restarts via orchestrator policies.

## 11. Monitoring, Logging, and Observability

Metrics:
- HTTP, latency, and error metrics via Prometheus exporter.
- Trend pipeline KPIs: ingested posts/min, avg inference time, drift alerts/day.

Logging:
- Structured logs (JSON recommended for production).
- Correlation IDs per request/job (recommended extension).

Alerting:
- Critical risk events and ingestion stalls can integrate with Slack/PagerDuty.

## 12. CI/CD Design

GitHub Actions pipeline:
- Backend dependency install + tests.
- Frontend dependency install + build.
- Extend with:
  - container build/push
  - SAST/DAST
  - signed image promotion
  - progressive deployment (canary)

## 13. Production-Readiness Checklist

- [x] Authentication and protected APIs
- [x] Modular service boundaries
- [x] Real-time event streaming
- [x] Model registry and drift check APIs
- [x] Risk decision engine
- [x] Monitoring stack integration
- [x] Docker/K8s manifests
- [x] CI workflow baseline
- [ ] API key vault integration (cloud secret manager)
- [ ] WAF + API gateway policies
- [ ] PostgreSQL migration for HA writes
- [ ] Full disaster recovery runbook

## 14. Future Expansion Roadmap

1. Add YouTube and RapidAPI review connectors with adaptive source weighting.
2. Replace heuristic summarization with LLM summarizer + guardrails.
3. Add multilingual sentiment with mBERT and language-specific calibration.
4. Introduce temporal transformers (TFT/Informer) for long-seasonality prediction.
5. Add recommendation engine with causal features (price, inventory, campaign exposure).
6. Move to event-driven microservices (Kafka + Flink + feature store).
7. Implement enterprise policy engine for compliance and PII controls.
8. Add FinOps and autoscaling optimization with workload forecasting.
