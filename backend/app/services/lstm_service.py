from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from flask import current_app


def _to_feature_matrix(metrics: List) -> np.ndarray:
    return np.array(
        [
            [
                float(item.mentions),
                float(item.avg_sentiment),
                float(item.engagement),
            ]
            for item in metrics
        ],
        dtype=np.float32,
    )


def _build_sequences(normalized: np.ndarray, sequence_length: int):
    x_arr = []
    y_arr = []
    for idx in range(len(normalized) - sequence_length):
        x_arr.append(normalized[idx : idx + sequence_length])
        y_arr.append(normalized[idx + sequence_length, :2])
    return np.array(x_arr), np.array(y_arr)


def _train_runtime_model(metrics: List) -> Optional[dict]:
    if len(metrics) < 10:
        return None

    try:
        import torch
        import torch.nn as nn
    except Exception:
        return None

    sequence_length = current_app.config.get("LSTM_SEQUENCE_LENGTH", 6)
    epochs = current_app.config.get("LSTM_EPOCHS", 25)
    hidden_size = current_app.config.get("LSTM_HIDDEN_SIZE", 32)
    learning_rate = current_app.config.get("LSTM_LEARNING_RATE", 0.01)
    if len(metrics) <= sequence_length + 2:
        return None

    class TrendLSTM(nn.Module):
        def __init__(self, in_features: int = 3, hidden: int = 32):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=in_features, hidden_size=hidden, batch_first=True
            )
            self.head = nn.Sequential(
                nn.Linear(hidden, max(8, hidden // 2)),
                nn.ReLU(),
                nn.Linear(max(8, hidden // 2), 2),
            )

        def forward(self, batch):
            out, _ = self.lstm(batch)
            return self.head(out[:, -1, :])

    torch.manual_seed(42)
    features = _to_feature_matrix(metrics)
    mu = features.mean(axis=0)
    sigma = features.std(axis=0)
    sigma[sigma == 0] = 1.0
    normalized = (features - mu) / sigma

    x_arr, y_arr = _build_sequences(normalized, sequence_length)
    if len(x_arr) < 6:
        return None

    x = torch.tensor(x_arr, dtype=torch.float32)
    y = torch.tensor(y_arr, dtype=torch.float32)

    split = max(4, int(len(x) * 0.8))
    x_train, x_val = x[:split], x[split:]
    y_train, y_val = y[:split], y[split:]
    if len(x_val) == 0:
        x_val, y_val = x_train[-2:], y_train[-2:]

    model = TrendLSTM(hidden=hidden_size)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    model.train()
    for _ in range(max(5, int(epochs))):
        pred = model(x_train)
        loss = criterion(pred, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        val_pred = model(x_val)
        val_loss = float(criterion(val_pred, y_val).item())
        final_window = torch.tensor(
            normalized[-sequence_length:].reshape(1, sequence_length, 3),
            dtype=torch.float32,
        )
        next_pred = model(final_window).cpu().numpy()[0]

    forecast_mentions = float(next_pred[0] * sigma[0] + mu[0])
    forecast_sentiment = float(next_pred[1] * sigma[1] + mu[1])
    confidence = float(max(0.1, min(0.95, 1.0 / (1.0 + val_loss))))

    return {
        "forecast_mentions": max(0.0, forecast_mentions),
        "forecast_sentiment": float(np.clip(forecast_sentiment, -1.0, 1.0)),
        "confidence": confidence,
        "method": "lstm_runtime",
    }


def _load_trained_model_forecast(metrics: List) -> Optional[dict]:
    try:
        import torch
        import torch.nn as nn
    except Exception:
        return None

    if len(metrics) < 8:
        return None

    backend_root = Path(__file__).resolve().parents[2]
    model_path = backend_root / current_app.config.get(
        "LSTM_MODEL_PATH", "models/lstm_trend.pt"
    )
    if not model_path.exists():
        return None

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    sequence_length = int(checkpoint.get("sequence_length", 6))
    hidden_size = int(checkpoint.get("hidden_size", 32))
    mu = np.array(checkpoint.get("mu", [0.0, 0.0, 0.0]), dtype=np.float32)
    sigma = np.array(checkpoint.get("sigma", [1.0, 1.0, 1.0]), dtype=np.float32)
    sigma[sigma == 0] = 1.0

    class TrendLSTM(nn.Module):
        def __init__(self, in_features: int = 3, hidden: int = 32):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=in_features, hidden_size=hidden, batch_first=True
            )
            self.head = nn.Sequential(
                nn.Linear(hidden, max(8, hidden // 2)),
                nn.ReLU(),
                nn.Linear(max(8, hidden // 2), 2),
            )

        def forward(self, batch):
            out, _ = self.lstm(batch)
            return self.head(out[:, -1, :])

    model = TrendLSTM(hidden=hidden_size)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    features = _to_feature_matrix(metrics)
    normalized = (features - mu) / sigma
    if len(normalized) < sequence_length:
        return None

    with torch.no_grad():
        window = torch.tensor(
            normalized[-sequence_length:].reshape(1, sequence_length, 3),
            dtype=torch.float32,
        )
        next_pred = model(window).numpy()[0]

    forecast_mentions = float(next_pred[0] * sigma[0] + mu[0])
    forecast_sentiment = float(next_pred[1] * sigma[1] + mu[1])
    return {
        "forecast_mentions": max(0.0, forecast_mentions),
        "forecast_sentiment": float(np.clip(forecast_sentiment, -1.0, 1.0)),
        "confidence": float(checkpoint.get("train_confidence", 0.6)),
        "method": "lstm_trained",
    }


def forecast_with_lstm(metrics: List) -> Optional[Dict]:
    trained = _load_trained_model_forecast(metrics)
    if trained:
        return trained
    return _train_runtime_model(metrics)
