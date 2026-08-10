from datetime import datetime, timedelta
from typing import Dict, List

import numpy as np
from flask import current_app
from sqlalchemy import func

from app.extensions import db
from app.models import Product, ProductMetric, SocialPost, TrendForecast
from app.services.lstm_service import forecast_with_lstm


def _hour_bucket(dt: datetime) -> datetime:
    return dt.replace(minute=0, second=0, microsecond=0)


def aggregate_product_metrics(product_id: int, window_hours: int = 24) -> List[ProductMetric]:
    since = datetime.utcnow() - timedelta(hours=window_hours)
    rows = (
        db.session.query(
            func.strftime("%Y-%m-%d %H:00:00", SocialPost.created_ts).label("bucket"),
            func.avg(SocialPost.sentiment_score).label("avg_sentiment"),
            func.count(SocialPost.id).label("mentions"),
            func.sum(SocialPost.engagement_score).label("engagement"),
        )
        .filter(SocialPost.product_id == product_id, SocialPost.created_ts >= since)
        .group_by("bucket")
        .order_by("bucket")
        .all()
    )

    metrics = []
    for row in rows:
        interval_start = datetime.strptime(row.bucket, "%Y-%m-%d %H:%M:%S")
        mentions = int(row.mentions or 0)
        avg_sent = float(row.avg_sentiment or 0.0)
        engagement = float(row.engagement or 0.0)
        trend_score = float((mentions * 0.45) + ((avg_sent + 1) * 30) + (engagement * 0.02))

        metric = ProductMetric.query.filter_by(
            product_id=product_id, interval_start=interval_start, interval_minutes=60
        ).first()
        if not metric:
            metric = ProductMetric(
                product_id=product_id, interval_start=interval_start, interval_minutes=60
            )
            db.session.add(metric)

        metric.avg_sentiment = avg_sent
        metric.mentions = mentions
        metric.engagement = engagement
        metric.trend_score = trend_score
        metrics.append(metric)

    db.session.commit()
    return metrics


def forecast_product_trend(product_id: int, horizon_hours: int = 24) -> TrendForecast:
    metrics = (
        ProductMetric.query.filter_by(product_id=product_id)
        .order_by(ProductMetric.interval_start.asc())
        .all()
    )

    if not metrics:
        forecast_mentions, forecast_sentiment, confidence = 0.0, 0.0, 0.0
    else:
        lstm_result = None
        if current_app.config.get("ENABLE_LSTM_MODEL", True):
            lstm_result = forecast_with_lstm(metrics)

        if lstm_result:
            forecast_mentions = float(lstm_result["forecast_mentions"])
            forecast_sentiment = float(lstm_result["forecast_sentiment"])
            confidence = float(lstm_result["confidence"])
        else:
            mention_series = np.array([m.mentions for m in metrics], dtype=float)
            sentiment_series = np.array([m.avg_sentiment for m in metrics], dtype=float)
            last_mentions = float(mention_series[-1])
            recent_growth = (
                float(np.diff(mention_series[-4:]).mean())
                if len(mention_series) > 4
                else 0.0
            )
            forecast_mentions = max(0.0, last_mentions + recent_growth)
            forecast_sentiment = (
                float(sentiment_series[-3:].mean())
                if len(sentiment_series) >= 3
                else float(sentiment_series[-1])
            )
            volatility = (
                float(np.std(sentiment_series[-12:]))
                if len(sentiment_series) > 2
                else 0.4
            )
            confidence = max(0.1, min(0.95, 1.0 - volatility))

    row = TrendForecast.query.filter_by(product_id=product_id, forecast_horizon_hours=horizon_hours).first()
    if not row:
        row = TrendForecast(product_id=product_id, forecast_horizon_hours=horizon_hours)
        db.session.add(row)

    row.forecast_mentions = forecast_mentions
    row.forecast_sentiment = forecast_sentiment
    row.confidence = confidence
    db.session.commit()
    return row


def list_trending_products(
    limit: int = 10,
    category: str | None = None,
    start_dt: datetime | None = None,
    end_dt: datetime | None = None,
) -> List[Dict]:
    metric_filter = []
    if start_dt and end_dt:
        metric_filter.extend(
            [ProductMetric.interval_start >= start_dt, ProductMetric.interval_start < end_dt]
        )

    latest_metrics = (
        db.session.query(ProductMetric.product_id, func.max(ProductMetric.interval_start).label("latest"))
        .filter(*metric_filter)
        .group_by(ProductMetric.product_id)
        .subquery()
    )

    query = (
        db.session.query(
            Product.display_name,
            Product.slug,
            Product.category,
            ProductMetric.trend_score,
            ProductMetric.avg_sentiment,
            ProductMetric.mentions,
        )
        .join(latest_metrics, latest_metrics.c.product_id == Product.id)
        .join(
            ProductMetric,
            (ProductMetric.product_id == latest_metrics.c.product_id)
            & (ProductMetric.interval_start == latest_metrics.c.latest),
        )
    )
    if category:
        query = query.filter(func.lower(Product.category) == category.lower())

    rows = query.order_by(ProductMetric.trend_score.desc()).limit(limit).all()

    return [
        {
            "product": row.display_name,
            "slug": row.slug,
            "category": row.category or "other",
            "trend_score": round(float(row.trend_score), 2),
            "avg_sentiment": round(float(row.avg_sentiment), 3),
            "mentions": int(row.mentions),
        }
        for row in rows
    ]
