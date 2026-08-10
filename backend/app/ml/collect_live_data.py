import argparse
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[2]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app import create_app
from app.services.ingestion_service import ingest_product_data


def run_collection(queries: list[str]):
    app = create_app({"TESTING": True})
    with app.app_context():
        for query in queries:
            out = ingest_product_data(query)
            print(
                f"{query}: fetched={out.get('fetched_posts')} inserted={out.get('inserted_posts')} "
                f"cleaned={out.get('removed_noisy_posts')}"
            )


def main():
    parser = argparse.ArgumentParser(description="Collect live social data into local dataset.")
    parser.add_argument(
        "--queries",
        default="iPhone 15,Samsung S24,OnePlus 12,Bajaj CT 110 X ES",
        help="Comma-separated product queries.",
    )
    args = parser.parse_args()
    queries = [q.strip() for q in args.queries.split(",") if q.strip()]
    run_collection(queries)


if __name__ == "__main__":
    main()
