import argparse
import json
from datetime import datetime

from app import create_app
from app.services.model_lifecycle import register_model


def main():
    parser = argparse.ArgumentParser(description="Register/fine-tune sentiment model metadata.")
    parser.add_argument("--version", default=datetime.utcnow().strftime("%Y%m%d%H%M"))
    parser.add_argument("--artifact-uri", default="s3://ml-artifacts/sentiment/model.bin")
    parser.add_argument("--f1", type=float, default=0.91)
    parser.add_argument("--accuracy", type=float, default=0.93)
    args = parser.parse_args()

    app = create_app()
    with app.app_context():
        row = register_model(
            model_name="bert-sentiment",
            model_version=args.version,
            metrics={"f1": args.f1, "accuracy": args.accuracy, "task": "sentiment-classification"},
            artifact_uri=args.artifact_uri,
        )
        print(json.dumps({"id": row.id, "model_name": row.model_name, "model_version": row.model_version}))


if __name__ == "__main__":
    main()
