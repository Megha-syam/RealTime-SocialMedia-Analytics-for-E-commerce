import json
from datetime import datetime
from typing import Dict

from app.extensions import db
from app.models import ModelRegistry, SocialPost


def register_model(model_name: str, model_version: str, metrics: Dict, artifact_uri: str = "") -> ModelRegistry:
    row = ModelRegistry(
        model_name=model_name,
        model_version=model_version,
        status="active",
        metrics_json=json.dumps(metrics),
        artifact_uri=artifact_uri,
    )
    db.session.add(row)
    db.session.commit()
    return row


def list_models() -> list[Dict]:
    rows = ModelRegistry.query.order_by(ModelRegistry.created_at.desc()).all()
    out = []
    for row in rows:
        out.append(
            {
                "id": row.id,
                "model_name": row.model_name,
                "model_version": row.model_version,
                "status": row.status,
                "metrics": json.loads(row.metrics_json) if row.metrics_json else {},
                "artifact_uri": row.artifact_uri,
                "created_at": row.created_at.isoformat(),
            }
        )
    return out


def monitor_drift(product_id: int) -> Dict:
    posts = SocialPost.query.filter_by(product_id=product_id).order_by(SocialPost.created_ts.desc()).limit(200).all()
    if len(posts) < 30:
        return {"drift_detected": False, "reason": "Not enough samples"}

    newest = [p.sentiment_score for p in posts[:100]]
    baseline = [p.sentiment_score for p in posts[100:200]]
    diff = abs((sum(newest) / len(newest)) - (sum(baseline) / len(baseline)))

    return {
        "drift_detected": diff > 0.25,
        "drift_score": round(diff, 3),
        "checked_at": datetime.utcnow().isoformat(),
    }
