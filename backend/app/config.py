import os
from datetime import timedelta


class Config:
    APP_NAME = "rse-analytics"
    SECRET_KEY = os.getenv(
        "SECRET_KEY",
        "dev-secret-key-change-me-please-min-32-chars",
    )
    JWT_SECRET_KEY = os.getenv(
        "JWT_SECRET_KEY",
        "dev-jwt-secret-key-change-me-please-min-32-chars",
    )
    JWT_ACCESS_TOKEN_EXPIRES = timedelta(hours=6)
    SQLALCHEMY_DATABASE_URI = os.getenv("DATABASE_URL", "sqlite:///rse_analytics.db")
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    INGESTION_INTERVAL_SECONDS = int(os.getenv("INGESTION_INTERVAL_SECONDS", "60"))
    MAX_POSTS_PER_SOURCE = int(os.getenv("MAX_POSTS_PER_SOURCE", "30"))
    CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*")
    ENABLE_PROMETHEUS = os.getenv("ENABLE_PROMETHEUS", "true").lower() == "true"
    NLP_MODEL = os.getenv("NLP_MODEL", "distilbert-base-uncased-finetuned-sst-2-english")
    LOCAL_BERT_MODEL_PATH = os.getenv("LOCAL_BERT_MODEL_PATH", "models/bert_sentiment")
    BERT_LABELS_PATH = os.getenv("BERT_LABELS_PATH", "models/bert_sentiment/labels.json")
    LOCAL_DATASET_PATH = os.getenv("LOCAL_DATASET_PATH", "data/social_posts.csv")
    USE_LIVE_SOURCES = os.getenv("USE_LIVE_SOURCES", "true").lower() == "true"
    ENABLE_MOCK_DATA = os.getenv("ENABLE_MOCK_DATA", "false").lower() == "true"
    ENABLE_GEMINI = os.getenv("ENABLE_GEMINI", "true").lower() == "true"
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
    GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
    GEMINI_TIMEOUT_SECONDS = int(os.getenv("GEMINI_TIMEOUT_SECONDS", "20"))

    # LSTM settings for temporal trend forecast.
    ENABLE_LSTM_MODEL = os.getenv("ENABLE_LSTM_MODEL", "true").lower() == "true"
    LSTM_SEQUENCE_LENGTH = int(os.getenv("LSTM_SEQUENCE_LENGTH", "6"))
    LSTM_EPOCHS = int(os.getenv("LSTM_EPOCHS", "25"))
    LSTM_HIDDEN_SIZE = int(os.getenv("LSTM_HIDDEN_SIZE", "32"))
    LSTM_LEARNING_RATE = float(os.getenv("LSTM_LEARNING_RATE", "0.01"))
    LSTM_MODEL_PATH = os.getenv("LSTM_MODEL_PATH", "models/lstm_trend.pt")
