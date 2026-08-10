import json
from pathlib import Path
from typing import Dict

import torch

from app.models import ModelRegistry


def _backend_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _from_registry(model_name_hint: str) -> Dict:
    row = (
        ModelRegistry.query.filter(ModelRegistry.model_name.ilike(f"%{model_name_hint}%"))
        .order_by(ModelRegistry.created_at.desc())
        .first()
    )
    if not row:
        return {}
    metrics = {}
    if row.metrics_json:
        try:
            metrics = json.loads(row.metrics_json)
        except Exception:
            metrics = {}
    return {
        "metrics": metrics,
        "created_at": row.created_at.isoformat(),
        "model_version": row.model_version,
    }


def _bert_artifact_metrics() -> Dict:
    labels_path = _backend_root() / "models" / "bert_sentiment" / "labels.json"
    if not labels_path.exists():
        return {}
    try:
        with labels_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return {
            "accuracy": float(payload.get("val_accuracy", 0.0)),
            "live_samples": int(payload.get("live_samples", 0)),
            "queries": payload.get("queries", []),
            "source": "artifact",
        }
    except Exception:
        return {}


def _lstm_artifact_metrics() -> Dict:
    model_path = _backend_root() / "models" / "lstm_trend.pt"
    if not model_path.exists():
        return {}
    try:
        payload = torch.load(model_path, map_location="cpu")
        return {
            "confidence": float(payload.get("train_confidence", 0.0)),
            "live_rows": int(payload.get("live_rows", 0)),
            "trained_on": payload.get("trained_on", ""),
            "source": "artifact",
        }
    except Exception:
        return {}


def get_model_metrics() -> Dict:
    bert = _bert_artifact_metrics()
    if not bert:
        reg = _from_registry("bert")
        metrics = reg.get("metrics", {})
        bert = {
            "accuracy": float(metrics.get("val_accuracy", metrics.get("accuracy", 0.0))),
            "live_samples": int(metrics.get("live_samples", 0)),
            "source": "registry" if reg else "unavailable",
            "created_at": reg.get("created_at"),
            "model_version": reg.get("model_version"),
        }

    lstm = _lstm_artifact_metrics()
    if not lstm:
        reg = _from_registry("lstm")
        metrics = reg.get("metrics", {})
        lstm = {
            "confidence": float(metrics.get("confidence", metrics.get("accuracy", 0.0))),
            "live_rows": int(metrics.get("live_rows", 0)),
            "source": "registry" if reg else "unavailable",
            "created_at": reg.get("created_at"),
            "model_version": reg.get("model_version"),
        }

    return {"bert": bert, "lstm": lstm}
