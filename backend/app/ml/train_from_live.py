import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[2]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.ml.collect_live_data import run_collection
from app.ml.train_bert_local import main as train_bert_main
from app.ml.train_lstm_local import main as train_lstm_main


def main():
    # 1) Pull latest live rows into local dataset.
    run_collection(["iPhone 15", "Samsung S24", "OnePlus 12", "Bajaj CT 110 X ES"])
    # 2) Train BERT with base + live rows.
    train_bert_main()
    # 3) Train LSTM using live time-series (fallback to base dataset if needed).
    train_lstm_main()
    print("Live-data training pipeline complete.")


if __name__ == "__main__":
    main()
