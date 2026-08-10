from datetime import datetime
from typing import Dict

from app.extensions import db, socketio
from app.models import Product, RiskEvent, SocialPost
from app.services.category_service import classify_product_category
from app.services.connectors import fetch_all_sources
from app.services.gemini_service import generate_search_insight
from app.services.nlp_service import score_sentiment
from app.services.risk_engine import evaluate_post_risk, evaluate_product_risk
from app.services.trend_service import aggregate_product_metrics, forecast_product_trend
from app.utils.relevance import is_off_topic, is_product_mention
from app.utils.text import clean_text, detect_language


def _slugify(text: str) -> str:
    return "-".join(clean_text(text).split())


def get_or_create_product(query: str) -> Product:
    slug = _slugify(query)
    product = Product.query.filter_by(slug=slug).first()
    if not product:
        category = classify_product_category(query)
        product = Product(slug=slug, display_name=query.strip(), category=category)
        db.session.add(product)
        db.session.commit()
    elif not product.category or product.category.strip().lower() == "other":
        category = classify_product_category(query)
        if category and category != "other":
            product.category = category
            db.session.commit()
    return product


def ingest_product_data(query: str, include_ai: bool = True) -> Dict:
    product = get_or_create_product(query)
    rows = fetch_all_sources(query)
    source_counts = {}
    for row in rows:
        source = row.get("source", "unknown")
        source_counts[source] = source_counts.get(source, 0) + 1

    inserted = 0
    batch_seen_texts = set()

    for row in rows:
        if not row.get("text"):
            continue
        if not is_product_mention(row["text"], query):
            continue
        if is_off_topic(row["text"]):
            continue
        normalized_text = clean_text(row["text"])
        if not normalized_text or normalized_text in batch_seen_texts:
            continue
        batch_seen_texts.add(normalized_text)

        sentiment = score_sentiment(row["text"])
        risk = evaluate_post_risk(
            sentiment_score=sentiment["score"],
            engagement=float(row.get("engagement_score", 0.0)),
            text=row["text"],
        )

        existing = SocialPost.query.filter_by(
            source=row["source"], external_id=row["external_id"]
        ).first()
        if existing:
            continue

        post = SocialPost(
            product_id=product.id,
            source=row["source"],
            external_id=row["external_id"],
            author=row.get("author", "unknown"),
            text=row["text"],
            language=detect_language(row["text"]),
            created_ts=row.get("created_ts") or datetime.utcnow(),
            engagement_score=float(row.get("engagement_score", 0.0)),
            sentiment_score=float(sentiment["score"]),
            sentiment_label=str(sentiment["label"]),
            risk_flag=risk.severity if risk else None,
        )
        db.session.add(post)
        inserted += 1

        if risk:
            db.session.add(
                RiskEvent(
                    product_id=product.id,
                    severity=risk.severity,
                    trigger=risk.trigger,
                    details=risk.details,
                )
            )

    db.session.commit()

    # Cleanup historical noisy rows for the same product so trend/sentiment stay meaningful.
    existing_posts = SocialPost.query.filter_by(product_id=product.id).all()
    removed = 0
    for old in existing_posts:
        if (
            (old.source or "").lower() in {"local", "mock", "synthetic"}
            or not is_product_mention(old.text, query)
            or is_off_topic(old.text)
        ):
            db.session.delete(old)
            removed += 1
    if removed:
        db.session.commit()

    metrics = aggregate_product_metrics(product.id, window_hours=48)
    forecast = forecast_product_trend(product.id, horizon_hours=24)

    mention_growth = 0.0
    if len(metrics) > 1:
        mention_growth = metrics[-1].mentions - metrics[-2].mentions
    product_risk = evaluate_product_risk(
        avg_sentiment=(metrics[-1].avg_sentiment if metrics else 0.0),
        mention_growth=float(mention_growth),
    )
    if product_risk:
        db.session.add(
            RiskEvent(
                product_id=product.id,
                severity=product_risk.severity,
                trigger=product_risk.trigger,
                details=product_risk.details,
            )
        )
        db.session.commit()

    event_payload = {
        "product": product.display_name,
        "slug": product.slug,
        "category": product.category or "other",
        "fetched_posts": len(rows),
        "source_counts": source_counts,
        "inserted_posts": inserted,
        "removed_noisy_posts": removed,
        "forecast_mentions": round(float(forecast.forecast_mentions), 2),
        "forecast_sentiment": round(float(forecast.forecast_sentiment), 3),
        "confidence": round(float(forecast.confidence), 2),
    }
    if inserted == 0:
        event_payload["warning"] = (
            "No new unique rows were ingested in this cycle. "
            "If live sources are enabled, try broader product keywords."
        )
    if include_ai:
        ai_brief = generate_search_insight(
            query=query,
            category=product.category or "other",
            source_counts=source_counts,
            forecast={
                "forecast_mentions": event_payload["forecast_mentions"],
                "forecast_sentiment": event_payload["forecast_sentiment"],
                "confidence": event_payload["confidence"],
            },
        )
        if ai_brief:
            event_payload["ai_search_insight"] = ai_brief
    socketio.emit("analytics_update", event_payload, namespace="/stream")

    return event_payload
