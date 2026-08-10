from datetime import datetime
from typing import Dict

from sqlalchemy import func

from app.extensions import db
from app.models import Product, ProductMetric
from app.services.gemini_service import generate_comparison_reason
from app.services.ingestion_service import ingest_product_data
from app.services.summary_service import generate_product_summary
from app.utils.text import clean_text


def _slugify(text: str) -> str:
    return "-".join(clean_text(text).split())


def _product_snapshot(
    slug: str,
    start_dt: datetime | None = None,
    end_dt: datetime | None = None,
) -> Dict:
    product = Product.query.filter_by(slug=slug).first()
    if not product:
        raise ValueError(f"Product with slug '{slug}' not found.")

    metric_query = ProductMetric.query.filter_by(product_id=product.id)
    avg_query = db.session.query(func.avg(ProductMetric.avg_sentiment)).filter(
        ProductMetric.product_id == product.id
    )
    mention_query = db.session.query(func.sum(ProductMetric.mentions)).filter(
        ProductMetric.product_id == product.id
    )
    if start_dt and end_dt:
        metric_query = metric_query.filter(
            ProductMetric.interval_start >= start_dt,
            ProductMetric.interval_start < end_dt,
        )
        avg_query = avg_query.filter(
            ProductMetric.interval_start >= start_dt,
            ProductMetric.interval_start < end_dt,
        )
        mention_query = mention_query.filter(
            ProductMetric.interval_start >= start_dt,
            ProductMetric.interval_start < end_dt,
        )

    latest = metric_query.order_by(ProductMetric.interval_start.desc()).first()
    avg_sentiment = avg_query.scalar()
    mention_sum = mention_query.scalar()

    return {
        "product": product.display_name,
        "slug": product.slug,
        "category": product.category or "other",
        "sentiment": round(float(avg_sentiment or 0.0), 3),
        "trend_score": round(float(latest.trend_score if latest else 0.0), 2),
        "mentions": int(mention_sum or 0),
    }


def compare_products(
    product_a: str,
    product_b: str,
    start_dt: datetime | None = None,
    end_dt: datetime | None = None,
) -> Dict:
    ingest_product_data(product_a, include_ai=False)
    ingest_product_data(product_b, include_ai=False)
    slug_a = _slugify(product_a)
    slug_b = _slugify(product_b)
    snap_a = _product_snapshot(slug_a, start_dt=start_dt, end_dt=end_dt)
    snap_b = _product_snapshot(slug_b, start_dt=start_dt, end_dt=end_dt)

    winner = snap_a if snap_a["trend_score"] >= snap_b["trend_score"] else snap_b
    base_recommendation = (
        f"{winner['product']} currently has stronger social momentum and better trend score."
    )
    recommendation = generate_comparison_reason(snap_a, snap_b, base_recommendation)
    deltas = {
        "sentiment_delta_left_minus_right": round(
            float(snap_a["sentiment"] - snap_b["sentiment"]), 3
        ),
        "trend_score_delta_left_minus_right": round(
            float(snap_a["trend_score"] - snap_b["trend_score"]), 2
        ),
        "mentions_delta_left_minus_right": int(snap_a["mentions"] - snap_b["mentions"]),
    }
    winner_by_dimension = {
        "sentiment": snap_a["product"] if snap_a["sentiment"] >= snap_b["sentiment"] else snap_b["product"],
        "trend_score": snap_a["product"] if snap_a["trend_score"] >= snap_b["trend_score"] else snap_b["product"],
        "mentions": snap_a["product"] if snap_a["mentions"] >= snap_b["mentions"] else snap_b["product"],
    }

    return {
        "left": snap_a,
        "right": snap_b,
        "recommended": winner["product"],
        "recommendation_reason": recommendation,
        "comparison_summary": recommendation,
        "deltas": deltas,
        "winner_by_dimension": winner_by_dimension,
    }


def compare_with_summaries(
    product_a: str,
    product_b: str,
    window_minutes: int = 43200,
    start_dt: datetime | None = None,
    end_dt: datetime | None = None,
) -> Dict:
    result = compare_products(product_a, product_b, start_dt=start_dt, end_dt=end_dt)
    for key in ["left", "right"]:
        product = Product.query.filter_by(slug=result[key]["slug"]).first()
        result[key]["summary"] = generate_product_summary(
            product.id,
            sample_size=120,
            window_minutes=window_minutes,
            start_dt=start_dt,
            end_dt=end_dt,
        )
    return result
