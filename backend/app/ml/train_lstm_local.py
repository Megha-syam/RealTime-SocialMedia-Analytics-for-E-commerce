from datetime import datetime
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn as nn

BACKEND_ROOT = Path(__file__).resolve().parents[2]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app import create_app
from app.services.connectors import fetch_live_for_queries


class TrendLSTM(nn.Module):
    def __init__(self, in_features: int = 3, hidden: int = 32):
        super().__init__()
        self.lstm = nn.LSTM(input_size=in_features, hidden_size=hidden, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden, max(8, hidden // 2)),
            nn.ReLU(),
            nn.Linear(max(8, hidden // 2), 2),
        )

    def forward(self, batch):
        out, _ = self.lstm(batch)
        return self.head(out[:, -1, :])


def _label_from_text(text: str) -> float:
    normalized = text.lower()
    pos = ["excellent", "great", "good", "smooth", "fast", "improved", "reliable"]
    neg = ["drain", "heat", "issue", "slow", "delay", "problem", "expensive"]
    p = sum(1 for w in pos if w in normalized)
    n = sum(1 for w in neg if w in normalized)
    if p + n == 0:
        return 0.0
    return float((p - n) / (p + n))


def _build_live_series(bucket_minutes: int = 10):
    queries = ["iPhone 15", "Samsung S24", "OnePlus 12", "Bajaj CT 110 X ES"]
    app = create_app({"TESTING": True, "USE_LIVE_SOURCES": True})
    with app.app_context():
        rows = fetch_live_for_queries(queries, limit_per_query=120)

    buckets = {}
    for row in rows:
        ts = row.get("created_ts", datetime.utcnow())
        minute = (ts.minute // bucket_minutes) * bucket_minutes
        bucket = ts.replace(minute=minute, second=0, microsecond=0)
        key = bucket.isoformat()
        if key not in buckets:
            buckets[key] = {"mentions": 0, "sentiment_sum": 0.0, "engagement": 0.0}
        buckets[key]["mentions"] += 1
        buckets[key]["sentiment_sum"] += _label_from_text(row.get("text", ""))
        buckets[key]["engagement"] += float(row.get("engagement_score", 1.0))

    features = []
    for key in sorted(buckets.keys()):
        row = buckets[key]
        avg_sent = row["sentiment_sum"] / max(1, row["mentions"])
        features.append([float(row["mentions"]), float(avg_sent), float(row["engagement"])])
    return np.array(features, dtype=np.float32), len(rows)


def _build_sequences(normalized: np.ndarray, sequence_length: int):
    x_arr = []
    y_arr = []
    for idx in range(len(normalized) - sequence_length):
        x_arr.append(normalized[idx : idx + sequence_length])
        y_arr.append(normalized[idx + sequence_length, :2])
    return np.array(x_arr), np.array(y_arr)


def main():
    model_dir = BACKEND_ROOT / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "lstm_trend.pt"

    features, row_count = _build_live_series(bucket_minutes=10)
    if len(features) < 16:
        raise RuntimeError("Not enough live time-series points for LSTM training.")

    sequence_length = 6
    hidden_size = 32
    epochs = 100

    mu = features.mean(axis=0)
    sigma = features.std(axis=0)
    sigma[sigma == 0] = 1.0
    normalized = (features - mu) / sigma

    x_arr, y_arr = _build_sequences(normalized, sequence_length)
    x = torch.tensor(x_arr, dtype=torch.float32)
    y = torch.tensor(y_arr, dtype=torch.float32)

    split = max(4, int(len(x) * 0.8))
    x_train, x_val = x[:split], x[split:]
    y_train, y_val = y[:split], y[split:]
    if len(x_val) == 0:
        x_val, y_val = x_train[-2:], y_train[-2:]

    model = TrendLSTM(hidden=hidden_size)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    model.train()
    for _ in range(epochs):
        pred = model(x_train)
        loss = criterion(pred, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        val_loss = float(criterion(model(x_val), y_val).item())
    confidence = float(max(0.1, min(0.95, 1.0 / (1.0 + val_loss))))

    torch.save(
        {
            "state_dict": model.state_dict(),
            "sequence_length": sequence_length,
            "hidden_size": hidden_size,
            "mu": mu.tolist(),
            "sigma": sigma.tolist(),
            "train_confidence": confidence,
            "trained_on": "live_links",
            "live_rows": row_count,
        },
        model_path,
    )
    print(
        f"Saved trained LSTM model to {model_path} "
        f"(confidence={confidence:.3f}, live_rows={row_count})"
    )


if __name__ == "__main__":
    main()
