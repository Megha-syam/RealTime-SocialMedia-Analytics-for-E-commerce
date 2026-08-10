from collections import Counter
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Dict, List
import re

from sqlalchemy import desc

from app.models import Product, SocialPost
from app.services.gemini_service import enhance_summary
from app.services.nlp_service import score_sentiment
from app.utils.relevance import is_commerce_context, is_off_topic, is_product_mention
from app.utils.text import clean_text

URL_RE = re.compile(r"https?://\S+")
HTML_TAG_RE = re.compile(r"<[^>]+>")
REVIEW_SOURCES = {"reddit", "twitter", "youtube", "review", "reviews", "ecommerce"}
ASPECT_KEYWORDS = {
    "camera quality": {"camera", "photo", "photos", "video", "videos", "portrait"},
    "battery life": {"battery", "drain", "backup", "charging", "charge"},
    "performance": {"performance", "smooth", "fast", "lag", "gaming", "fps"},
    "display": {"display", "screen", "brightness", "bezel"},
    "software experience": {"software", "update", "ui", "bug", "crash"},
    "build and design": {"design", "build", "premium", "compact", "weight"},
    "price and value": {"price", "cost", "value", "sale", "discount"},
    "delivery and service": {"delivery", "service", "support", "warranty", "return"},
    "heating": {"heating", "overheat", "hot", "temperature"},
    "network and connectivity": {"network", "signal", "5g", "wifi"},
}


@dataclass
class _InstantPost:
    source: str
    text: str
    sentiment_label: str
    sentiment_score: float
    engagement_score: float
    created_ts: datetime


def _query_recent_posts(
    product_id: int,
    sample_size: int,
    window_minutes: int,
    start_dt: datetime | None = None,
    end_dt: datetime | None = None,
):
    query = SocialPost.query.filter(SocialPost.product_id == product_id)
    if start_dt and end_dt:
        query = query.filter(SocialPost.created_ts >= start_dt, SocialPost.created_ts < end_dt)
    else:
        cutoff = datetime.utcnow() - timedelta(minutes=window_minutes)
        query = query.filter(SocialPost.created_ts >= cutoff)
    return query.order_by(desc(SocialPost.created_ts)).limit(sample_size).all()


def _short_text(text: str, max_len: int = 200) -> str:
    raw = text or ""
    raw = HTML_TAG_RE.sub(" ", raw)
    raw = URL_RE.sub(" ", raw)
    compact = " ".join(raw.split())
    if len(compact) <= max_len:
        return compact
    return compact[: max_len - 3].rstrip() + "..."


def _post_label_and_score(post: SocialPost) -> tuple[str, float]:
    label = (post.sentiment_label or "").lower().strip()
    score = float(post.sentiment_score or 0.0)

    # Guardrail: explicit complaint terms should not be reported as positive.
    normalized = clean_text(post.text)
    complaint_terms = {
        "issue",
        "problem",
        "delay",
        "drain",
        "heating",
        "overheat",
        "refund",
        "defect",
        "broken",
        "slow",
    }
    complaint_resolved_terms = {
        "no issue",
        "without issue",
        "issue fixed",
        "problem solved",
        "resolved",
    }
    if (
        any(term in normalized for term in complaint_terms)
        and not any(term in normalized for term in complaint_resolved_terms)
        and label != "negative"
    ):
        return "negative", -max(abs(score), 0.35)

    if label in {"positive", "neutral", "negative"}:
        return label, score
    inferred = score_sentiment(post.text)
    return str(inferred["label"]), float(inferred["score"])


def _top_examples(posts: List[SocialPost], label: str, top_k: int = 6) -> List[SocialPost]:
    candidates = []
    for post in posts:
        post_label, score = _post_label_and_score(post)
        if post_label != label:
            continue
        weight = abs(score) * max(1.0, float(post.engagement_score or 0.0))
        candidates.append((weight, post))

    candidates.sort(
        key=lambda item: (
            item[0],
            item[1].created_ts or datetime.min,
        ),
        reverse=True,
    )

    selected = []
    seen = set()
    for _, post in candidates:
        key = clean_text(post.text)
        if not key or key in seen:
            continue
        seen.add(key)
        selected.append(post)
        if len(selected) >= top_k:
            break
    return selected


def _build_evidence_paragraph(posts: List[SocialPost], tone: str) -> str:
    if not posts:
        if tone == "positive":
            return "No stable positive theme is visible in this live window."
        return "No stable negative complaint pattern is visible in this live window."

    def topic_stub(text: str) -> str:
        trimmed = _short_text(text, 130)
        for sep in [" - ", "|", ":", "…", "..."]:
            if sep in trimmed:
                trimmed = trimmed.split(sep)[0]
        words = trimmed.split()
        if len(words) > 12:
            trimmed = " ".join(words[:12])
        return trimmed.strip(" .,-")

    stubs = []
    seen = set()
    for post in posts[:3]:
        stub = topic_stub(post.text)
        key = clean_text(stub)
        if not key or key in seen:
            continue
        seen.add(key)
        stubs.append(stub)

    if not stubs:
        if tone == "positive":
            return "No stable positive theme is visible in this live window."
        return "No stable negative complaint pattern is visible in this live window."

    if len(stubs) == 1:
        key_points = stubs[0]
    elif len(stubs) == 2:
        key_points = f"{stubs[0]} and {stubs[1]}"
    else:
        key_points = f"{stubs[0]}, {stubs[1]}, and {stubs[2]}"

    if tone == "positive":
        return (
            "In the selected period, positive conversations are mainly about "
            f"{key_points}."
        )
    return (
        "In the selected period, negative conversations are mainly about "
        f"{key_points}."
    )


def _select_review_posts(posts: List[SocialPost]) -> List[SocialPost]:
    community = [p for p in posts if (p.source or "").lower() in REVIEW_SOURCES]
    if len(community) >= 4:
        return community
    non_trend = [p for p in posts if (p.source or "").lower() != "google_trends"]
    return non_trend if non_trend else posts


def _aspect_counts(posts: List[SocialPost], label: str) -> Counter:
    counts: Counter = Counter()
    for post in posts:
        post_label, _ = _post_label_and_score(post)
        if post_label != label:
            continue
        text = clean_text(post.text)
        for aspect, keywords in ASPECT_KEYWORDS.items():
            if any(keyword in text for keyword in keywords):
                counts[aspect] += 1
    return counts


def _build_review_paragraph(posts: List[SocialPost], tone: str) -> str:
    if not posts:
        if tone == "positive":
            return "No stable positive review pattern is visible in this live window."
        return "No stable negative complaint pattern is visible in this live window."

    counts = _aspect_counts(posts, "positive" if tone == "positive" else "negative")
    top_aspects = [name for name, count in counts.most_common(3) if count > 0]
    if top_aspects:
        if len(top_aspects) == 1:
            aspects = top_aspects[0]
        elif len(top_aspects) == 2:
            aspects = f"{top_aspects[0]} and {top_aspects[1]}"
        else:
            aspects = f"{top_aspects[0]}, {top_aspects[1]}, and {top_aspects[2]}"

        if tone == "positive":
            return (
                "Review conversations are mostly positive, with users appreciating "
                f"{aspects}."
            )
        return (
            "Review conversations show recurring concerns around "
            f"{aspects}."
        )

    # Fallback to concise narrative if aspect matching is weak.
    excerpt = _short_text(posts[0].text, 120)
    if tone == "positive":
        return f"Review conversations are mostly positive; key feedback highlights {excerpt}."
    return f"Review conversations show concerns such as {excerpt}."


def _summarize_from_posts(posts: List[SocialPost], window_minutes: int) -> Dict[str, object]:
    if not posts:
        return {
            "pros": [],
            "cons": [],
            "pros_keyphrases": [],
            "cons_keyphrases": [],
            "pros_paragraph": "No sufficient live positive evidence yet.",
            "cons_paragraph": "No sufficient live negative evidence yet.",
            "overall": "Insufficient live data for this time window.",
        }

    labels = []
    scores = []
    for post in posts:
        label, score = _post_label_and_score(post)
        labels.append(label)
        scores.append(score)

    counts = Counter(labels)
    total = len(posts)
    pos = counts.get("positive", 0)
    neg = counts.get("negative", 0)
    neu = counts.get("neutral", 0)
    avg = (sum(scores) / len(scores)) if scores else 0.0

    if avg > 0.12 and pos >= neg:
        stance = "positive with active purchase interest"
    elif avg < -0.12 and neg >= pos:
        stance = "negative with recurring concerns"
    else:
        stance = "mixed and still moving"

    overall = (
        f"Based on {total} live mentions in the last {window_minutes} minutes "
        f"({pos} positive, {neg} negative, {neu} neutral), sentiment is {stance}."
    )

    review_posts = _select_review_posts(posts)
    top_pos = _top_examples(review_posts, "positive", top_k=6)
    top_neg = _top_examples(review_posts, "negative", top_k=6)

    pros = [_short_text(post.text, 190) for post in top_pos]
    cons = [_short_text(post.text, 190) for post in top_neg]

    return {
        "pros": pros,
        "cons": cons,
        "pros_keyphrases": [],
        "cons_keyphrases": [],
        "pros_paragraph": _build_review_paragraph(top_pos, "positive"),
        "cons_paragraph": _build_review_paragraph(top_neg, "negative"),
        "overall": overall,
    }


def generate_product_summary(
    product_id: int,
    sample_size: int = 200,
    window_minutes: int = 43200,
    start_dt: datetime | None = None,
    end_dt: datetime | None = None,
    ) -> Dict:
    posts = _query_recent_posts(
        product_id,
        sample_size,
        window_minutes,
        start_dt=start_dt,
        end_dt=end_dt,
    )
    # Prefer real ingested rows; ignore historical mock rows when possible.
    real_posts = [p for p in posts if "-mock-" not in (p.external_id or "")]
    if real_posts:
        posts = real_posts

    # Enforce real external evidence only; ignore legacy local/mock fixtures.
    posts = [
        p
        for p in posts
        if (p.source or "").lower() not in {"local", "mock", "synthetic"}
    ]

    product = Product.query.get(product_id)
    product_name = product.display_name if product else ""
    normalized_product = clean_text(product_name)
    if normalized_product:
        posts = [p for p in posts if is_product_mention(p.text, product_name)]

    # Never use off-topic rows in summary.
    posts = [p for p in posts if not is_off_topic(p.text)]

    # Prefer commerce-context posts when enough signal is present.
    commerce_posts = [p for p in posts if is_commerce_context(p.text)]
    if len(commerce_posts) >= 4:
        posts = commerce_posts

    summary = _summarize_from_posts(posts, window_minutes=window_minutes)
    summary["recommendation_paragraph"] = (
        "Use trend score + sentiment trajectory together before making pricing or campaign decisions."
    )

    llm_summary = enhance_summary(
        product_name=product_name or "product",
        base_summary=summary,
        pros_examples=summary.get("pros", []),
        cons_examples=summary.get("cons", []),
    )
    if llm_summary:
        summary["overall"] = llm_summary.get("overall") or summary["overall"]
        summary["pros_paragraph"] = llm_summary.get("pros_paragraph") or summary["pros_paragraph"]
        summary["cons_paragraph"] = llm_summary.get("cons_paragraph") or summary["cons_paragraph"]
        summary["recommendation_paragraph"] = (
            llm_summary.get("recommendation_paragraph")
            or summary["recommendation_paragraph"]
        )
        summary["ai_model"] = "gemini"
    else:
        summary["ai_model"] = "deterministic"

    summary["sample_size"] = len(posts)
    summary["window_minutes"] = window_minutes
    summary["signal_quality"] = "high" if len(posts) >= 20 else "medium" if len(posts) >= 8 else "low"
    summary["source_breakdown"] = {}
    for post in posts:
        summary["source_breakdown"][post.source] = (
            summary["source_breakdown"].get(post.source, 0) + 1
        )
    return summary


def _coerce_instant_posts(reviews_input) -> List[_InstantPost]:
    if reviews_input is None:
        return []

    raw_items: List = []
    if isinstance(reviews_input, str):
        raw_items = [line.strip() for line in reviews_input.splitlines() if line.strip()]
    elif isinstance(reviews_input, dict):
        raw_items = [reviews_input]
    elif isinstance(reviews_input, list):
        raw_items = reviews_input
    else:
        return []

    rows: List[_InstantPost] = []
    for item in raw_items:
        if isinstance(item, str):
            text = item.strip()
            payload = {}
        elif isinstance(item, dict):
            text = str(item.get("text") or item.get("review") or "").strip()
            payload = item
        else:
            continue

        if not text:
            continue

        sentiment_label = str(
            payload.get("sentiment_label") or payload.get("sentiment") or ""
        ).strip().lower()
        try:
            sentiment_score = float(payload.get("sentiment_score", 0.0))
        except (TypeError, ValueError):
            sentiment_score = 0.0
        try:
            engagement_score = float(payload.get("engagement_score", 1.0))
        except (TypeError, ValueError):
            engagement_score = 1.0

        rows.append(
            _InstantPost(
                source=str(payload.get("source", "instant_input")),
                text=text,
                sentiment_label=sentiment_label,
                sentiment_score=sentiment_score,
                engagement_score=engagement_score,
                created_ts=datetime.utcnow(),
            )
        )
    return rows


def generate_instant_summary(
    product_name: str,
    reviews_input,
    window_minutes: int = 43200,
) -> Dict:
    posts = _coerce_instant_posts(reviews_input)
    summary = _summarize_from_posts(posts, window_minutes=window_minutes)
    summary["recommendation_paragraph"] = (
        "Use trend score + sentiment trajectory together before making pricing or campaign decisions."
    )

    llm_summary = enhance_summary(
        product_name=product_name or "product",
        base_summary=summary,
        pros_examples=summary.get("pros", []),
        cons_examples=summary.get("cons", []),
    )
    if llm_summary:
        summary["overall"] = llm_summary.get("overall") or summary["overall"]
        summary["pros_paragraph"] = llm_summary.get("pros_paragraph") or summary["pros_paragraph"]
        summary["cons_paragraph"] = llm_summary.get("cons_paragraph") or summary["cons_paragraph"]
        summary["recommendation_paragraph"] = (
            llm_summary.get("recommendation_paragraph")
            or summary["recommendation_paragraph"]
        )
        summary["ai_model"] = "gemini"
    else:
        summary["ai_model"] = "deterministic"

    summary["sample_size"] = len(posts)
    summary["window_minutes"] = window_minutes
    summary["signal_quality"] = "high" if len(posts) >= 20 else "medium" if len(posts) >= 8 else "low"
    summary["source_breakdown"] = {}
    for post in posts:
        summary["source_breakdown"][post.source] = (
            summary["source_breakdown"].get(post.source, 0) + 1
        )
    return summary
