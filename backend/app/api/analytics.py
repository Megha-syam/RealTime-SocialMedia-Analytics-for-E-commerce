from datetime import datetime, timedelta

from flask import Blueprint, jsonify, request
from flask_jwt_extended import jwt_required
from sqlalchemy import desc

from app.models import Product, ProductMetric, RiskEvent, SocialPost
from app.services.comparison_service import compare_with_summaries
from app.services.gemini_service import (
    generate_dashboard_insight,
    generate_original_reviews,
    list_allowed_categories,
    normalize_category,
)
from app.services.ingestion_service import ingest_product_data
from app.services.model_lifecycle import list_models, monitor_drift, register_model
from app.services.model_metrics_service import get_model_metrics
from app.services.summary_service import generate_instant_summary, generate_product_summary
from app.services.trend_service import list_trending_products
from app.utils.text import clean_text

analytics_bp = Blueprint("analytics", __name__)


def _slugify(text: str) -> str:
    return "-".join(clean_text(text).split())


def _period_bounds(year: int | None, month: int | None) -> tuple[datetime | None, datetime | None]:
    if not year:
        return None, None
    if month and 1 <= month <= 12:
        start = datetime(year, month, 1)
        if month == 12:
            end = datetime(year + 1, 1, 1)
        else:
            end = datetime(year, month + 1, 1)
        return start, end
    return datetime(year, 1, 1), datetime(year + 1, 1, 1)


@analytics_bp.post("/products/search")
@jwt_required()
def search_product():
    payload = request.get_json(silent=True) or {}
    query = (payload.get("query") or "").strip()
    if not query:
        return jsonify({"error": "query is required"}), 400
    result = ingest_product_data(query, include_ai=True)
    return jsonify(result), 200


@analytics_bp.post("/products/reviews/original")
@jwt_required()
def product_original_reviews():
    payload = request.get_json(silent=True) or {}
    product_name = (payload.get("product") or payload.get("query") or "").strip()
    category = (payload.get("category") or "").strip()

    if not product_name:
        return jsonify({"error": "product is required"}), 400
    if not category:
        return jsonify({"error": "category is required"}), 400
    if not normalize_category(category):
        return (
            jsonify(
                {
                    "error": "invalid category",
                    "allowed_categories": list_allowed_categories(),
                }
            ),
            400,
        )

    try:
        requested_count = int(payload.get("count", 5))
    except (TypeError, ValueError):
        return jsonify({"error": "count must be an integer"}), 400

    result = generate_original_reviews(
        product_name=product_name,
        category=category,
        count=requested_count,
    )
    if not result:
        return (
            jsonify(
                {
                    "error": "unable to generate reviews from Gemini API",
                    "allowed_categories": list_allowed_categories(),
                }
            ),
            503,
        )
    return jsonify(result), 200


@analytics_bp.get("/products/<slug>/dashboard")
@jwt_required()
def product_dashboard(slug: str):
    window_hours = int(request.args.get("window_hours", 8760))
    year = request.args.get("year", type=int)
    month = request.args.get("month", type=int)
    product = Product.query.filter_by(slug=slug).first()
    if not product:
        return jsonify({"error": "product not found"}), 404

    start_dt, end_dt = _period_bounds(year, month)
    if not start_dt:
        start_dt = datetime.utcnow() - timedelta(hours=window_hours)
        end_dt = datetime.utcnow()

    metrics = (
        ProductMetric.query.filter(
            ProductMetric.product_id == product.id,
            ProductMetric.interval_start >= start_dt,
            ProductMetric.interval_start < end_dt,
        )
        .order_by(ProductMetric.interval_start.asc())
        .all()
    )

    posts = (
        SocialPost.query.filter(
            SocialPost.product_id == product.id,
            SocialPost.created_ts >= start_dt,
            SocialPost.created_ts < end_dt,
        )
        .order_by(desc(SocialPost.created_ts))
        .limit(200)
        .all()
    )
    risk_events = (
        RiskEvent.query.filter(
            RiskEvent.product_id == product.id,
            RiskEvent.created_at >= start_dt,
            RiskEvent.created_at < end_dt,
        )
        .order_by(desc(RiskEvent.created_at))
        .limit(30)
        .all()
    )

    risk_payload = [
        {
            "severity": r.severity,
            "trigger": r.trigger,
            "details": r.details,
            "created_at": r.created_at.isoformat(),
        }
        for r in risk_events
    ]
    sentiment_dist = {
        "positive": len([p for p in posts if p.sentiment_label == "positive"]),
        "neutral": len([p for p in posts if p.sentiment_label == "neutral"]),
        "negative": len([p for p in posts if p.sentiment_label == "negative"]),
    }
    period_label = (
        f"{start_dt.strftime('%Y-%m-%d')} to {(end_dt - timedelta(seconds=1)).strftime('%Y-%m-%d')}"
        if year
        else f"last {window_hours} hours"
    )
    ai_insight = generate_dashboard_insight(
        product_name=product.display_name,
        period_label=period_label,
        sentiment_distribution=sentiment_dist,
        risk_events=risk_payload,
    )

    return jsonify(
        {
            "product": {
                "name": product.display_name,
                "slug": product.slug,
                "category": product.category or "other",
            },
            "period": {
                "start": start_dt.isoformat(),
                "end": end_dt.isoformat(),
                "year": year,
                "month": month,
            },
            "timeline": [
                {
                    "ts": row.interval_start.isoformat(),
                    "mentions": row.mentions,
                    "avg_sentiment": round(row.avg_sentiment, 3),
                    "engagement": round(row.engagement, 2),
                    "trend_score": round(row.trend_score, 2),
                }
                for row in metrics
            ],
            "sentiment_distribution": sentiment_dist,
            "risk_events": risk_payload,
            "recent_posts": [
                {
                    "source": p.source,
                    "text": p.text[:220],
                    "sentiment": p.sentiment_label,
                    "score": round(p.sentiment_score, 3),
                    "engagement": round(p.engagement_score, 2),
                    "created_ts": p.created_ts.isoformat(),
                }
                for p in posts[:20]
            ],
            "ai_dashboard_insight": ai_insight,
            "model_metrics": get_model_metrics(),
        }
    )


@analytics_bp.get("/products/<slug>/summary")
@jwt_required()
def product_summary(slug: str):
    product = Product.query.filter_by(slug=slug).first()
    if not product:
        return jsonify({"error": "product not found"}), 404
    # Pull latest posts before summarizing to keep output real-time.
    ingest_product_data(product.display_name, include_ai=False)
    sample_size = int(request.args.get("sample_size", 250))
    window_minutes = int(request.args.get("window_minutes", 43200))
    year = request.args.get("year", type=int)
    month = request.args.get("month", type=int)
    start_dt, end_dt = _period_bounds(year, month)
    summary = generate_product_summary(
        product.id,
        sample_size=sample_size,
        window_minutes=window_minutes,
        start_dt=start_dt,
        end_dt=end_dt,
    )
    return jsonify(
        {
            "product": product.display_name,
            "category": product.category or "other",
            "summary": summary,
            "period": {"year": year, "month": month},
        }
    ), 200


@analytics_bp.post("/products/summary/instant")
@jwt_required()
def product_summary_instant():
    payload = request.get_json(silent=True) or {}
    product_name = (payload.get("product") or payload.get("query") or "").strip()
    if not product_name:
        return jsonify({"error": "product is required"}), 400

    reviews_input = payload.get("reviews")
    if reviews_input is None:
        reviews_input = payload.get("input")
    if reviews_input is None:
        reviews_input = payload.get("text")
    if reviews_input is None:
        return jsonify({"error": "reviews or input text is required"}), 400

    try:
        window_minutes = int(payload.get("window_minutes", 43200))
    except (TypeError, ValueError):
        return jsonify({"error": "window_minutes must be an integer"}), 400

    category = (payload.get("category") or "other").strip() or "other"
    summary = generate_instant_summary(
        product_name=product_name,
        reviews_input=reviews_input,
        window_minutes=window_minutes,
    )

    return (
        jsonify(
            {
                "product": product_name,
                "category": category,
                "summary": summary,
                "mode": "instant_input",
            }
        ),
        200,
    )


@analytics_bp.get("/products/trending")
@jwt_required()
def trending_products():
    limit = int(request.args.get("limit", 10))
    category = (request.args.get("category") or "").strip() or None
    year = request.args.get("year", type=int)
    month = request.args.get("month", type=int)
    start_dt, end_dt = _period_bounds(year, month)
    return jsonify(
        {
            "items": list_trending_products(
                limit=limit,
                category=category,
                start_dt=start_dt,
                end_dt=end_dt,
            )
        }
    ), 200


@analytics_bp.post("/products/compare")
@jwt_required()
def compare_products():
    payload = request.get_json(silent=True) or {}
    left = (payload.get("left") or "").strip()
    right = (payload.get("right") or "").strip()
    window_minutes = int(payload.get("window_minutes", 43200))
    year = int(payload.get("year") or 0) or None
    month = int(payload.get("month") or 0) or None
    if not left or not right:
        return jsonify({"error": "left and right products are required"}), 400
    start_dt, end_dt = _period_bounds(year, month)
    result = compare_with_summaries(
        left,
        right,
        window_minutes=window_minutes,
        start_dt=start_dt,
        end_dt=end_dt,
    )
    result["period"] = {"year": year, "month": month}
    return jsonify(result), 200


@analytics_bp.get("/products/<slug>/risks")
@jwt_required()
def list_product_risks(slug: str):
    product = Product.query.filter_by(slug=slug).first()
    if not product:
        return jsonify({"error": "product not found"}), 404
    rows = RiskEvent.query.filter_by(product_id=product.id).order_by(desc(RiskEvent.created_at)).limit(100).all()
    return jsonify(
        {
            "product": product.display_name,
            "risks": [
                {
                    "severity": r.severity,
                    "trigger": r.trigger,
                    "details": r.details,
                    "resolved": r.resolved,
                    "created_at": r.created_at.isoformat(),
                }
                for r in rows
            ],
        }
    )


@analytics_bp.get("/models")
@jwt_required()
def models_status():
    return jsonify({"models": list_models()}), 200


@analytics_bp.get("/models/metrics")
@jwt_required()
def model_metrics():
    return jsonify({"metrics": get_model_metrics()}), 200


@analytics_bp.post("/models/register")
@jwt_required()
def model_register():
    payload = request.get_json(silent=True) or {}
    model_name = payload.get("model_name")
    model_version = payload.get("model_version")
    metrics = payload.get("metrics", {})
    artifact_uri = payload.get("artifact_uri", "")
    if not model_name or not model_version:
        return jsonify({"error": "model_name and model_version are required"}), 400
    row = register_model(model_name, model_version, metrics, artifact_uri)
    return jsonify({"id": row.id, "status": row.status}), 201


@analytics_bp.post("/models/drift/<slug>")
@jwt_required()
def model_drift(slug: str):
    product = Product.query.filter_by(slug=slug).first()
    if not product:
        return jsonify({"error": "product not found"}), 404
    result = monitor_drift(product.id)
    return jsonify(result), 200
