# API Reference (v1)

Base URL: `http://localhost:5000/api/v1`

All endpoints except `/health`, `/auth/register`, `/auth/login` require:
`Authorization: Bearer <JWT>`

## Health

`GET /health`

Response:
```json
{
  "status": "ok",
  "service": "rse-analytics",
  "realtime": true
}
```

## Auth

`POST /auth/register`
```json
{
  "full_name": "Demo User",
  "email": "demo@rse.ai",
  "password": "StrongPass123!"
}
```

`POST /auth/login`
```json
{
  "email": "demo@rse.ai",
  "password": "StrongPass123!"
}
```

## Search (Live Ingestion Trigger)

`POST /products/search`
```json
{
  "query": "iPhone 15"
}
```

## Dashboard

`GET /products/{slug}/dashboard?window_hours=48`

## Summary

`GET /products/{slug}/summary`

`POST /products/summary/instant`
```json
{
  "product": "iPhone 16 Pro",
  "category": "smartphone",
  "input": "Battery backup is strong\nCamera is excellent in low light",
  "window_minutes": 240
}
```

You can send either:
- `input` / `text` as plain text (line-by-line reviews), or
- `reviews` as an array of strings/objects.

## Original Reviews (Gemini Only)

`POST /products/reviews/original`
```json
{
  "product": "iPhone 16 Pro",
  "category": "smartphone",
  "count": 5
}
```

Response:
```json
{
  "product": "iPhone 16 Pro",
  "category": "smartphone",
  "source": "gemini",
  "reviews": [
    {
      "sentiment": "positive",
      "rating": 5,
      "review": "..."
    }
  ]
}
```

## Compare

`POST /products/compare`
```json
{
  "left": "iPhone 15",
  "right": "Samsung S24"
}
```

## Trending

`GET /products/trending?limit=10`

## Risk Events

`GET /products/{slug}/risks`

## Model Lifecycle

`GET /models`

`POST /models/register`
```json
{
  "model_name": "bert-sentiment",
  "model_version": "20260222-01",
  "metrics": {
    "f1": 0.92,
    "accuracy": 0.94
  },
  "artifact_uri": "s3://ml-artifacts/sentiment/20260222-01"
}
```

`POST /models/drift/{slug}`
