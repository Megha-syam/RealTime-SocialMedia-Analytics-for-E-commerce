from datetime import datetime

from .extensions import db


class TimestampMixin:
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    updated_at = db.Column(
        db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )


class User(TimestampMixin, db.Model):
    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(255), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255), nullable=False)
    full_name = db.Column(db.String(255), nullable=False)
    role = db.Column(db.String(32), nullable=False, default="analyst")
    is_active = db.Column(db.Boolean, default=True, nullable=False)


class Product(TimestampMixin, db.Model):
    __tablename__ = "products"

    id = db.Column(db.Integer, primary_key=True)
    slug = db.Column(db.String(120), unique=True, nullable=False, index=True)
    display_name = db.Column(db.String(255), nullable=False)
    category = db.Column(db.String(100), nullable=True)


class SearchSession(TimestampMixin, db.Model):
    __tablename__ = "search_sessions"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False, index=True)
    product_id = db.Column(db.Integer, db.ForeignKey("products.id"), nullable=False, index=True)
    query_text = db.Column(db.String(255), nullable=False)
    source_count = db.Column(db.Integer, nullable=False, default=0)


class SocialPost(TimestampMixin, db.Model):
    __tablename__ = "social_posts"

    id = db.Column(db.Integer, primary_key=True)
    product_id = db.Column(db.Integer, db.ForeignKey("products.id"), nullable=False, index=True)
    source = db.Column(db.String(40), nullable=False, index=True)
    external_id = db.Column(db.String(255), nullable=False)
    author = db.Column(db.String(120), nullable=True)
    text = db.Column(db.Text, nullable=False)
    language = db.Column(db.String(10), nullable=True, default="en")
    created_ts = db.Column(db.DateTime, nullable=False, index=True)
    engagement_score = db.Column(db.Float, nullable=False, default=0.0)
    sentiment_score = db.Column(db.Float, nullable=False, default=0.0)
    sentiment_label = db.Column(db.String(16), nullable=False, default="neutral")
    risk_flag = db.Column(db.String(32), nullable=True)

    __table_args__ = (
        db.UniqueConstraint("source", "external_id", name="uq_post_source_external"),
    )


class ProductMetric(TimestampMixin, db.Model):
    __tablename__ = "product_metrics"

    id = db.Column(db.Integer, primary_key=True)
    product_id = db.Column(db.Integer, db.ForeignKey("products.id"), nullable=False, index=True)
    interval_start = db.Column(db.DateTime, nullable=False, index=True)
    interval_minutes = db.Column(db.Integer, nullable=False, default=60)
    avg_sentiment = db.Column(db.Float, nullable=False, default=0.0)
    mentions = db.Column(db.Integer, nullable=False, default=0)
    engagement = db.Column(db.Float, nullable=False, default=0.0)
    trend_score = db.Column(db.Float, nullable=False, default=0.0)

    __table_args__ = (
        db.UniqueConstraint(
            "product_id", "interval_start", "interval_minutes", name="uq_product_interval"
        ),
    )


class TrendForecast(TimestampMixin, db.Model):
    __tablename__ = "trend_forecasts"

    id = db.Column(db.Integer, primary_key=True)
    product_id = db.Column(db.Integer, db.ForeignKey("products.id"), nullable=False, index=True)
    forecast_horizon_hours = db.Column(db.Integer, nullable=False, default=24)
    forecast_mentions = db.Column(db.Float, nullable=False, default=0.0)
    forecast_sentiment = db.Column(db.Float, nullable=False, default=0.0)
    confidence = db.Column(db.Float, nullable=False, default=0.0)


class RiskEvent(TimestampMixin, db.Model):
    __tablename__ = "risk_events"

    id = db.Column(db.Integer, primary_key=True)
    product_id = db.Column(db.Integer, db.ForeignKey("products.id"), nullable=False, index=True)
    severity = db.Column(db.String(16), nullable=False, index=True)
    trigger = db.Column(db.String(120), nullable=False)
    details = db.Column(db.Text, nullable=False)
    resolved = db.Column(db.Boolean, default=False, nullable=False)


class ModelRegistry(TimestampMixin, db.Model):
    __tablename__ = "model_registry"

    id = db.Column(db.Integer, primary_key=True)
    model_name = db.Column(db.String(120), nullable=False, index=True)
    model_version = db.Column(db.String(64), nullable=False)
    status = db.Column(db.String(24), nullable=False, default="active")
    metrics_json = db.Column(db.Text, nullable=True)
    artifact_uri = db.Column(db.String(500), nullable=True)
