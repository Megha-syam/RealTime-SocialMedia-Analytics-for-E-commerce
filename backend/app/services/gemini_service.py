import json
import re
from typing import Dict, Iterable, List

import requests
from flask import current_app


JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)
CATEGORY_ALIASES = {
    "mobile": "smartphone",
    "smart phone": "smartphone",
    "phone": "smartphone",
    "cell phone": "smartphone",
    "television": "tv",
    "smart tv": "tv",
    "earbuds": "wearable",
    "headphones": "wearable",
    "smartwatch": "wearable",
    "watch": "wearable",
    "vehicle": "automotive",
    "car": "automotive",
    "motorcycle": "bike",
}
ALLOWED_CATEGORIES = {
    "smartphone",
    "laptop",
    "tablet",
    "tv",
    "wearable",
    "automotive",
    "bike",
    "appliance",
    "gaming",
    "other",
}
ALLOWED_REVIEW_SENTIMENTS = {"positive", "neutral", "negative"}


def _is_enabled() -> bool:
    return bool(
        current_app.config.get("ENABLE_GEMINI", True)
        and current_app.config.get("GEMINI_API_KEY", "").strip()
    )


def _extract_json(text: str) -> Dict | None:
    if not text:
        return None
    raw = text.strip()
    try:
        return json.loads(raw)
    except Exception:
        match = JSON_BLOCK_RE.search(raw)
        if not match:
            return None
        try:
            return json.loads(match.group(0))
        except Exception:
            return None


def _candidate_models() -> List[str]:
    configured = current_app.config.get("GEMINI_MODEL", "gemini-2.0-flash").strip()
    fallbacks = [
        "gemini-2.0-flash",
        "gemini-2.5-flash",
        "gemini-1.5-flash",
    ]
    ordered = [configured] + fallbacks
    seen = set()
    models: List[str] = []
    for model in ordered:
        if model and model not in seen:
            seen.add(model)
            models.append(model)
    return models


def _call_gemini(
    prompt: str,
    temperature: float = 0.2,
    max_tokens: int = 400,
    response_mime_type: str | None = None,
) -> str:
    if not _is_enabled():
        return ""

    api_key = current_app.config.get("GEMINI_API_KEY", "").strip()
    timeout = int(current_app.config.get("GEMINI_TIMEOUT_SECONDS", 20))
    last_error: Exception | None = None

    for model in _candidate_models():
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{model}:generateContent?key={api_key}"
        )
        generation_config = {
            "temperature": temperature,
            "maxOutputTokens": max_tokens,
        }
        if response_mime_type:
            generation_config["responseMimeType"] = response_mime_type

        payload = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": generation_config,
        }
        try:
            response = requests.post(url, json=payload, timeout=timeout)
            response.raise_for_status()
            data = response.json()
            candidates = data.get("candidates", [])
            if not candidates:
                return ""
            parts = candidates[0].get("content", {}).get("parts", [])
            text_parts = [p.get("text", "") for p in parts if p.get("text")]
            return "\n".join(text_parts).strip()
        except requests.HTTPError as exc:
            last_error = exc
            status = exc.response.status_code if exc.response is not None else None
            if status in {400, 404}:
                continue
            raise
        except requests.RequestException as exc:
            last_error = exc
            continue

    if last_error:
        raise last_error
    return ""


def _json_request(prompt: str, required_keys: Iterable[str], max_tokens: int = 420) -> Dict | None:
    try:
        text = _call_gemini(
            prompt,
            temperature=0.1,
            max_tokens=max_tokens,
            response_mime_type="application/json",
        )
    except Exception:
        return None
    payload = _extract_json(text)
    if not payload:
        return None
    if not all(key in payload for key in required_keys):
        return None
    return payload


def _normalize_category(raw_category: str) -> str:
    normalized = str(raw_category or "").strip().lower()
    normalized = re.sub(r"[^a-z\s]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    if normalized in ALLOWED_CATEGORIES:
        return normalized
    if normalized in CATEGORY_ALIASES:
        return CATEGORY_ALIASES[normalized]
    return ""


def normalize_category(raw_category: str) -> str:
    return _normalize_category(raw_category)


def list_allowed_categories() -> List[str]:
    return sorted(ALLOWED_CATEGORIES)


def classify_product_category(product_name: str) -> str:
    prompt = f"""
Classify this product into one category.
Product: "{product_name}"

Allowed categories:
smartphone, laptop, tablet, tv, wearable, automotive, bike, appliance, gaming, other

Return strict JSON:
{{"category":"one_of_allowed_categories"}}
""".strip()
    payload = _json_request(prompt, required_keys=["category"], max_tokens=80)
    if not payload:
        return ""
    return _normalize_category(payload.get("category", ""))


def generate_original_reviews(product_name: str, category: str, count: int = 5) -> Dict | None:
    normalized_category = _normalize_category(category)
    if not normalized_category:
        return None

    try:
        review_count = int(count or 5)
    except (TypeError, ValueError):
        review_count = 5
    review_count = max(1, min(review_count, 10))
    prompt = f"""
You are generating original user-style product reviews for test data creation.
Product: "{product_name}"
Category: "{normalized_category}"
Number of reviews: {review_count}

Rules:
- Return only strict JSON.
- Every review must be unique and original.
- Do not copy from known websites or include external links.
- Each review must be 25 to 55 words.
- Use sentiment values only from: positive, neutral, negative.
- Use rating from 1 to 5.

Return strict JSON:
{{
  "product": "{product_name}",
  "category": "{normalized_category}",
  "reviews": [
    {{"sentiment":"positive|neutral|negative","rating":1,"review":"text"}}
  ]
}}
""".strip()

    payload = _json_request(
        prompt,
        required_keys=["product", "category", "reviews"],
        max_tokens=max(320, review_count * 180),
    )
    if not payload:
        return None

    rows = payload.get("reviews")
    if not isinstance(rows, list):
        return None

    reviews: List[Dict] = []
    seen = set()
    for item in rows:
        if not isinstance(item, dict):
            continue
        text = str(item.get("review", "")).strip()
        if not text:
            continue
        dedupe_key = re.sub(r"\s+", " ", text.lower())
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)

        raw_sentiment = str(item.get("sentiment", "neutral")).strip().lower()
        sentiment = raw_sentiment if raw_sentiment in ALLOWED_REVIEW_SENTIMENTS else "neutral"
        try:
            rating = int(item.get("rating", 3))
        except (TypeError, ValueError):
            rating = 3
        rating = max(1, min(rating, 5))

        reviews.append(
            {
                "sentiment": sentiment,
                "rating": rating,
                "review": text,
            }
        )
        if len(reviews) >= review_count:
            break

    if not reviews:
        return None

    return {
        "product": product_name.strip(),
        "category": normalized_category,
        "source": "gemini",
        "reviews": reviews,
    }


def generate_search_insight(query: str, category: str, source_counts: Dict, forecast: Dict) -> str:
    prompt = f"""
You are an e-commerce intelligence analyst.
Create one concise paragraph (max 55 words) for live product search output.

Product: {query}
Category: {category or "unknown"}
Sources: {source_counts}
Forecast: {forecast}

Write direct operational insight for analysts. No bullet points.
""".strip()
    try:
        text = _call_gemini(prompt, temperature=0.2, max_tokens=120)
        return text if text else ""
    except Exception:
        return ""


def generate_dashboard_insight(product_name: str, period_label: str, sentiment_distribution: Dict, risk_events: List[Dict]) -> str:
    prompt = f"""
Generate one concise dashboard insight paragraph (max 65 words).
Product: {product_name}
Period: {period_label}
Sentiment distribution: {sentiment_distribution}
Recent risk events: {risk_events[:5]}

Focus on actionability for campaign/pricing/operations.
""".strip()
    try:
        text = _call_gemini(prompt, temperature=0.2, max_tokens=140)
        return text if text else ""
    except Exception:
        return ""


def generate_comparison_reason(left: Dict, right: Dict, base_reason: str) -> str:
    prompt = f"""
Create one concise comparison recommendation paragraph (max 70 words).
Left product snapshot: {left}
Right product snapshot: {right}
Current baseline reason: {base_reason}

Use only provided values and keep it deterministic/business-focused.
""".strip()
    try:
        text = _call_gemini(prompt, temperature=0.2, max_tokens=160)
        return text if text else base_reason
    except Exception:
        return base_reason


def enhance_summary(
    product_name: str,
    base_summary: Dict,
    pros_examples: List[str],
    cons_examples: List[str],
) -> Dict | None:
    prompt = f"""
You are a strict factual summarizer for e-commerce social analytics.
Rewrite summary using only provided evidence.

Product: {product_name}
Base overall: {base_summary.get("overall", "")}
Base recommendation: {base_summary.get("recommendation_paragraph", "")}
Pros examples: {pros_examples[:4]}
Cons examples: {cons_examples[:4]}

Return strict JSON with keys:
overall, pros_paragraph, cons_paragraph, recommendation_paragraph
""".strip()

    payload = _json_request(
        prompt,
        required_keys=[
            "overall",
            "pros_paragraph",
            "cons_paragraph",
            "recommendation_paragraph",
        ],
        max_tokens=260,
    )
    if not payload:
        return None
    return {
        "overall": str(payload.get("overall", "")).strip(),
        "pros_paragraph": str(payload.get("pros_paragraph", "")).strip(),
        "cons_paragraph": str(payload.get("cons_paragraph", "")).strip(),
        "recommendation_paragraph": str(payload.get("recommendation_paragraph", "")).strip(),
    }
