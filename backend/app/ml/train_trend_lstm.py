import argparse
import json
from datetime import datetime

import numpy as np

from app import create_app
from app.services.model_lifecycle import register_model


def fake_train(sequence_length: int, epochs: int) -> dict:
    # Placeholder training logic; replace with TensorFlow/PyTorch training pipeline.
    base = 0.8 + min(0.15, (epochs / 200))
    return {
        "mae": round(float(max(0.03, 0.25 - base * 0.2)), 4),
        "rmse": round(float(max(0.05, 0.35 - base * 0.25)), 4),
        "r2": round(float(min(0.98, base + np.random.uniform(0.01, 0.03))), 4),
        "sequence_length": sequence_length,
        "epochs": epochs,
    }


def main():
    parser = argparse.ArgumentParser(description="Train/register trend forecasting model.")
    parser.add_argument("--version", default=datetime.utcnow().strftime("%Y%m%d%H%M"))
    parser.add_argument("--sequence-length", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--artifact-uri", default="s3://ml-artifacts/trend-lstm/model.keras")
    args = parser.parse_args()

    metrics = fake_train(args.sequence_length, args.epochs)

    app = create_app()
    with app.app_context():
        row = register_model(
            model_name="lstm-trend-forecast",
            model_version=args.version,
            metrics=metrics,
            artifact_uri=args.artifact_uri,
        )
        print(json.dumps({"id": row.id, "metrics": metrics}))


if __name__ == "__main__":
    main()
