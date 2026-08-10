import argparse
import json

from app import create_app
from app.models import Product
from app.services.model_lifecycle import monitor_drift


def main():
    parser = argparse.ArgumentParser(description="Run drift check for a product slug")
    parser.add_argument("--slug", required=True)
    args = parser.parse_args()

    app = create_app()
    with app.app_context():
        product = Product.query.filter_by(slug=args.slug).first()
        if not product:
            print(json.dumps({"error": "product not found"}))
            return
        print(json.dumps(monitor_drift(product.id)))


if __name__ == "__main__":
    main()
