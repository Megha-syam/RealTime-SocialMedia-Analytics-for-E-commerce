from functools import lru_cache
from pathlib import Path
from typing import Dict, List

import numpy as np
from flask import current_app
from sklearn.feature_extraction.text import CountVectorizer

from app.utils.text import clean_text


@lru_cache(maxsize=1)
def _load_transformer_pipeline():
    try:
        from transformers import pipeline

        backend_root = Path(__file__).resolve().parents[2]
        local_model_path = backend_root / current_app.config.get(
            "LOCAL_BERT_MODEL_PATH", "models/bert_sentiment"
        )

        if local_model_path.exists():
            return pipeline("text-classification", model=str(local_model_path))

        model_name = current_app.config.get(
            "NLP_MODEL", "distilbert-base-uncased-finetuned-sst-2-english"
        )
        return pipeline("text-classification", model=model_name)
    except Exception:
        return None


def score_sentiment(text: str) -> Dict[str, float | str]:
    processed = clean_text(text)
    if not processed:
        return {"score": 0.0, "label": "neutral"}

    transformer = _load_transformer_pipeline()
    if transformer:
        result = transformer(processed[:512])[0]
        raw_score = float(result["score"])
        label = result["label"].lower()
        if any(tag in label for tag in ["positive", "label_2"]):
            signed = raw_score
        elif any(tag in label for tag in ["negative", "label_0"]):
            signed = -raw_score
        else:
            signed = 0.0
        return {"score": float(np.clip(signed, -1.0, 1.0)), "label": "positive" if signed > 0.1 else "negative" if signed < -0.1 else "neutral"}

    # Lightweight fallback when transformer dependencies are unavailable.
    positive_words = {"good", "great", "amazing", "best", "smooth", "fast", "love"}
    negative_words = {"bad", "worst", "slow", "drain", "heat", "issue", "problem", "sucks"}
    tokens = set(processed.split())
    pos_hits = len(tokens & positive_words)
    neg_hits = len(tokens & negative_words)
    score = 0.0 if (pos_hits + neg_hits) == 0 else (pos_hits - neg_hits) / (pos_hits + neg_hits)
    label = "positive" if score > 0.1 else "negative" if score < -0.1 else "neutral"
    return {"score": float(score), "label": label}


def _dedupe_texts(texts: List[str], top_k: int) -> List[str]:
    unique = []
    seen = set()
    for raw in texts:
        normalized = clean_text(raw)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        # Keep sentence style in UI but remove trailing punctuation duplication.
        sentence = raw.strip().replace("\n", " ").rstrip(".")
        sentence = sentence[:180].strip()
        unique.append(sentence)
        if len(unique) >= top_k:
            break
    return unique


def _build_paragraph(points: List[str], tone: str) -> str:
    if not points:
        return "Not enough evidence yet to form a stable insight."

    compact = [p[0].upper() + p[1:] if p else p for p in points]
    if len(compact) == 1:
        details = compact[0]
    elif len(compact) == 2:
        details = f"{compact[0]} and {compact[1]}"
    else:
        details = f"{', '.join(compact[:-1])}, and {compact[-1]}"

    if tone == "positive":
        return f"Users most frequently appreciate {details}."
    return f"Users most frequently report concerns around {details}."


def _extract_keyphrases(texts: List[str], top_k: int = 4) -> List[Dict[str, object]]:
    normalized = [clean_text(text) for text in texts if clean_text(text)]
    if not normalized:
        return []

    try:
        vectorizer = CountVectorizer(
            ngram_range=(2, 3),
            stop_words="english",
            token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9]+\b",
            max_features=400,
        )
        matrix = vectorizer.fit_transform(normalized)
        terms = vectorizer.get_feature_names_out()
        counts = np.asarray(matrix.sum(axis=0)).ravel()
        sorted_idx = counts.argsort()[::-1]

        phrases = []
        for idx in sorted_idx:
            phrase = terms[idx]
            count = int(counts[idx])
            if count <= 1:
                continue
            # Filter generic phrases with little business value.
            if phrase in {"product", "phone", "latest", "today", "update"}:
                continue
            phrases.append({"phrase": phrase, "count": count})
            if len(phrases) >= top_k:
                break
        return phrases
    except Exception:
        return []


def _build_paragraph_from_phrases(
    points: List[str], phrases: List[Dict[str, object]], tone: str
) -> str:
    if phrases:
        segments = [f"{item['phrase']} ({item['count']})" for item in phrases]
        if len(segments) == 1:
            details = segments[0]
        elif len(segments) == 2:
            details = f"{segments[0]} and {segments[1]}"
        else:
            details = f"{', '.join(segments[:-1])}, and {segments[-1]}"

        if tone == "positive":
            return f"In the latest live window, users highlight {details}."
        return f"In the latest live window, recurring complaints focus on {details}."

    if points:
        highlight = points[0]
        if tone == "positive":
            return (
                "In the latest live window, positive sentiment appears in mentions such as: "
                f"{highlight}."
            )
        return (
            "In the latest live window, negative sentiment appears in mentions such as: "
            f"{highlight}."
        )

    return _build_paragraph(points, tone)


def summarize_points(texts: List[str], top_k: int = 5) -> Dict[str, object]:
    if not texts:
        return {
            "pros": [],
            "cons": [],
            "pros_paragraph": "Not enough evidence yet to form a stable insight.",
            "cons_paragraph": "Not enough evidence yet to form a stable insight.",
            "overall": "Insufficient data for summary.",
        }

    scored = [{"text": t, **score_sentiment(t)} for t in texts]
    pros_raw = [s["text"] for s in scored if s["label"] == "positive"]
    cons_raw = [s["text"] for s in scored if s["label"] == "negative"]
    neutral_raw = [s["text"] for s in scored if s["label"] == "neutral"]
    pros = _dedupe_texts(pros_raw, top_k)
    cons = _dedupe_texts(cons_raw, top_k)
    neutral = _dedupe_texts(neutral_raw, top_k)
    pros_phrases = _extract_keyphrases(pros_raw, top_k=min(4, top_k))
    cons_phrases = _extract_keyphrases(cons_raw, top_k=min(4, top_k))

    avg = float(np.mean([s["score"] for s in scored])) if scored else 0.0
    if avg > 0.15:
        overall = "Overall sentiment is positive with strong user appreciation."
    elif avg < -0.15:
        overall = "Overall sentiment is negative with recurring user complaints."
    else:
        overall = "Overall sentiment is mixed and still evolving."

    pros_paragraph = _build_paragraph_from_phrases(pros, pros_phrases, "positive")
    cons_paragraph = _build_paragraph_from_phrases(cons, cons_phrases, "negative")
    if not pros and not cons:
        pros_paragraph = "Live mentions are mostly neutral; no stable positive theme detected yet."
        cons_paragraph = "Live mentions are mostly neutral; no stable negative theme detected yet."
    elif not pros:
        pros_paragraph = "Positive sentiment is currently weak; no recurring positive theme is stable yet."
    elif not cons:
        cons_paragraph = "Negative sentiment is currently weak; no recurring complaint pattern is stable yet."

    return {
        "pros": pros,
        "cons": cons,
        "pros_keyphrases": pros_phrases,
        "cons_keyphrases": cons_phrases,
        "pros_paragraph": pros_paragraph,
        "cons_paragraph": cons_paragraph,
        "overall": overall,
    }
