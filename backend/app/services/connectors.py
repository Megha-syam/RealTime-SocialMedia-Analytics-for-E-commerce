import hashlib
import os
from datetime import datetime
from html import unescape
import re
from typing import Dict, List
from urllib.parse import quote_plus
import xml.etree.ElementTree as ET

import requests
from flask import current_app

from app.utils.relevance import is_off_topic, is_product_mention
from app.utils.text import clean_text

HTML_TAG_RE = re.compile(r"<[^>]+>")


def _strip_html(text: str) -> str:
    return unescape(HTML_TAG_RE.sub(" ", text or "")).strip()


def _normalize_row(row: Dict, product: str) -> Dict | None:
    text = row.get("text", "")
    if not text:
        return None
    if not is_product_mention(text, product):
        return None
    if is_off_topic(text):
        return None

    normalized = clean_text(text)
    if not normalized:
        return None

    return {
        "source": row.get("source", "live"),
        "external_id": row.get("external_id")
        or hashlib.sha1((row.get("source", "") + normalized).encode("utf-8")).hexdigest(),
        "author": row.get("author", "unknown"),
        "text": text.strip(),
        "created_ts": row.get("created_ts") or datetime.utcnow(),
        "engagement_score": float(row.get("engagement_score", 1.0)),
    }


def _dedupe_rows(rows: List[Dict], limit: int, product: str) -> List[Dict]:
    filtered = []
    seen = set()
    for row in rows:
        item = _normalize_row(row, product)
        if not item:
            continue
        key = clean_text(item["text"])
        if key in seen:
            continue
        seen.add(key)
        filtered.append(item)
        if len(filtered) >= limit:
            break
    return filtered


def _query_variants(product: str) -> List[str]:
    normalized = clean_text(product)
    if not normalized:
        return [product]

    variants = [product]
    if "review" not in normalized:
        variants.append(f"{product} review")
    if "user review" not in normalized:
        variants.append(f"{product} user review")
    return variants


def _fetch_reddit_public(product: str, limit: int) -> List[Dict]:
    # Real link integration: https://www.reddit.com/search.json
    user_agent = os.getenv("REDDIT_USER_AGENT", "rse-analytics/1.0")
    url = "https://www.reddit.com/search.json"
    params = {"q": product, "sort": "new", "t": "week", "limit": min(limit, 100)}
    headers = {"User-Agent": user_agent}

    try:
        response = requests.get(url, params=params, headers=headers, timeout=15)
        response.raise_for_status()
        rows = []
        for child in response.json().get("data", {}).get("children", []):
            data = child.get("data", {})
            text = f"{data.get('title', '')} {data.get('selftext', '')}".strip()
            if not text:
                continue
            rows.append(
                {
                    "source": "reddit",
                    "external_id": data.get("id"),
                    "author": data.get("author") or "unknown",
                    "text": text,
                    "created_ts": datetime.utcfromtimestamp(
                        float(data.get("created_utc", datetime.utcnow().timestamp()))
                    ),
                    "engagement_score": float(
                        data.get("score", 0) + data.get("num_comments", 0)
                    ),
                }
            )
        return rows
    except Exception:
        return []


def _fetch_google_news(product: str, limit: int) -> List[Dict]:
    # Real link integration: https://news.google.com/rss/search?q=<query>
    query = quote_plus(product)
    url = f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        root = ET.fromstring(response.text)
        rows = []
        for item in root.findall(".//item"):
            title = _strip_html(item.findtext("title", default=""))
            description = _strip_html(item.findtext("description", default=""))
            text = f"{title} {description}".strip()
            if not text:
                continue
            link = item.findtext("link", default="")
            rows.append(
                {
                    "source": "google_news",
                    "external_id": hashlib.sha1(link.encode("utf-8")).hexdigest()
                    if link
                    else hashlib.sha1(text.encode("utf-8")).hexdigest(),
                    "author": "publisher",
                    "text": text,
                    "created_ts": datetime.utcnow(),
                    "engagement_score": 1.0,
                }
            )
        return rows[:limit]
    except Exception:
        return []


def _fetch_twitter(product: str, limit: int) -> List[Dict]:
    # Real link integration: https://api.twitter.com/2/tweets/search/recent
    bearer = os.getenv("TWITTER_BEARER_TOKEN")
    if not bearer:
        return []
    try:
        url = "https://api.twitter.com/2/tweets/search/recent"
        params = {
            "query": product,
            "max_results": min(limit, 100),
            "tweet.fields": "created_at,public_metrics,author_id",
        }
        headers = {"Authorization": f"Bearer {bearer}"}
        response = requests.get(url, params=params, headers=headers, timeout=15)
        response.raise_for_status()
        rows = []
        for tweet in response.json().get("data", []):
            metrics = tweet.get("public_metrics", {})
            rows.append(
                {
                    "source": "twitter",
                    "external_id": tweet.get("id"),
                    "author": tweet.get("author_id", "unknown"),
                    "text": tweet.get("text", ""),
                    "created_ts": datetime.fromisoformat(
                        tweet.get("created_at", "").replace("Z", "+00:00")
                    ).replace(tzinfo=None),
                    "engagement_score": float(
                        metrics.get("like_count", 0)
                        + metrics.get("retweet_count", 0)
                        + metrics.get("reply_count", 0)
                    ),
                }
            )
        return rows
    except Exception:
        return []


def _fetch_youtube_comments(product: str, limit: int) -> List[Dict]:
    # Real link integration:
    # search: https://www.googleapis.com/youtube/v3/search
    # comments: https://www.googleapis.com/youtube/v3/commentThreads
    api_key = os.getenv("YOUTUBE_API_KEY")
    if not api_key:
        return []

    search_url = "https://www.googleapis.com/youtube/v3/search"
    comments_url = "https://www.googleapis.com/youtube/v3/commentThreads"
    max_videos = max(1, min(20, limit // 5 if limit > 5 else 5))
    rows = []

    try:
        search_resp = requests.get(
            search_url,
            params={
                "part": "snippet",
                "q": product,
                "type": "video",
                "maxResults": max_videos,
                "key": api_key,
                "order": "date",
            },
            timeout=20,
        )
        search_resp.raise_for_status()
        videos = search_resp.json().get("items", [])

        for video in videos:
            video_id = video.get("id", {}).get("videoId")
            if not video_id:
                continue

            comments_resp = requests.get(
                comments_url,
                params={
                    "part": "snippet",
                    "videoId": video_id,
                    "maxResults": min(100, max(10, limit)),
                    "textFormat": "plainText",
                    "key": api_key,
                    "order": "time",
                },
                timeout=20,
            )

            if comments_resp.status_code != 200:
                continue

            for thread in comments_resp.json().get("items", []):
                snippet = (
                    thread.get("snippet", {})
                    .get("topLevelComment", {})
                    .get("snippet", {})
                )
                text = _strip_html(snippet.get("textDisplay", ""))
                if not text:
                    continue
                published = snippet.get("publishedAt", "")
                created_ts = (
                    datetime.fromisoformat(published.replace("Z", "+00:00")).replace(
                        tzinfo=None
                    )
                    if published
                    else datetime.utcnow()
                )
                like_count = float(snippet.get("likeCount", 0))
                rows.append(
                    {
                        "source": "youtube",
                        "external_id": thread.get("id"),
                        "author": snippet.get("authorDisplayName", "unknown"),
                        "text": text,
                        "created_ts": created_ts,
                        "engagement_score": like_count + 1.0,
                    }
                )
                if len(rows) >= limit:
                    return rows
        return rows
    except Exception:
        return []


def _fetch_pytrends_interest(product: str, limit: int) -> List[Dict]:
    # Optional live trend signal from Google Trends via pytrends.
    if os.getenv("ENABLE_PYTRENDS", "true").lower() != "true":
        return []

    try:
        from pytrends.request import TrendReq
    except Exception:
        return []

    try:
        hl = os.getenv("PYTRENDS_HL", "en-US")
        tz = int(os.getenv("PYTRENDS_TZ", "360"))
        geo = os.getenv("PYTRENDS_GEO", "US")
        timeframe = os.getenv("PYTRENDS_TIMEFRAME", "now 7-d")

        # retries must stay 0 for urllib3 v2 compatibility with current pytrends.
        client = TrendReq(hl=hl, tz=tz, timeout=(10, 20), retries=0, backoff_factor=0)
        client.build_payload([product], cat=0, timeframe=timeframe, geo=geo, gprop="")
        frame = client.interest_over_time()
        if frame is None or frame.empty or product not in frame.columns:
            return []

        rows = []
        points = frame[product].dropna().tail(max(5, min(limit, 72)))
        if points.empty:
            return []

        recent_avg = float(points.tail(min(6, len(points))).mean())
        for ts, value in points.items():
            interest = float(value)
            trend_word = "rising" if interest >= recent_avg else "cooling"
            created_ts = ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else datetime.utcnow()
            if created_ts.tzinfo is not None:
                created_ts = created_ts.replace(tzinfo=None)

            rows.append(
                {
                    "source": "google_trends",
                    "external_id": f"{clean_text(product)}-{created_ts.isoformat()}-pytrends",
                    "author": "google_trends",
                    "text": (
                        f"{product} Google Trends search interest is {trend_word} "
                        f"with index {int(interest)} in {geo or 'global'}."
                    ),
                    "created_ts": created_ts,
                    "engagement_score": max(1.0, interest),
                }
            )
            if len(rows) >= limit:
                break
        return rows
    except Exception:
        return []


def fetch_all_sources(product: str) -> List[Dict]:
    limit = int(current_app.config.get("MAX_POSTS_PER_SOURCE", 30))
    if not bool(current_app.config.get("USE_LIVE_SOURCES", True)):
        return []

    rows = []
    for query in _query_variants(product):
        rows.extend(_fetch_reddit_public(query, limit))
        rows.extend(_fetch_google_news(query, limit))
        rows.extend(_fetch_twitter(query, limit))
        rows.extend(_fetch_youtube_comments(query, limit))
    rows.extend(_fetch_pytrends_interest(product, limit))
    return _dedupe_rows(rows, limit * 5, product)


def fetch_live_for_queries(queries: List[str], limit_per_query: int = 30) -> List[Dict]:
    all_rows = []
    for query in queries:
        product_rows = []
        product_rows.extend(_fetch_reddit_public(query, limit_per_query))
        product_rows.extend(_fetch_google_news(query, limit_per_query))
        product_rows.extend(_fetch_twitter(query, limit_per_query))
        product_rows.extend(_fetch_youtube_comments(query, limit_per_query))
        product_rows.extend(_fetch_pytrends_interest(query, limit_per_query))
        all_rows.extend(_dedupe_rows(product_rows, limit_per_query * 5, query))
    return all_rows
