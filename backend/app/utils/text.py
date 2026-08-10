import re
from typing import List

from langdetect import LangDetectException, detect

URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
MENTION_PATTERN = re.compile(r"@\w+")
HASHTAG_PATTERN = re.compile(r"#\w+")
NON_ALNUM_PATTERN = re.compile(r"[^a-zA-Z0-9\s]")
MULTISPACE_PATTERN = re.compile(r"\s+")


def clean_text(text: str) -> str:
    cleaned = URL_PATTERN.sub(" ", text or "")
    cleaned = MENTION_PATTERN.sub(" ", cleaned)
    cleaned = HASHTAG_PATTERN.sub(" ", cleaned)
    cleaned = NON_ALNUM_PATTERN.sub(" ", cleaned)
    cleaned = MULTISPACE_PATTERN.sub(" ", cleaned).strip().lower()
    return cleaned


def detect_language(text: str) -> str:
    try:
        return detect(text)
    except LangDetectException:
        return "unknown"


def tokenize_keywords(query: str) -> List[str]:
    normalized = clean_text(query)
    return [token for token in normalized.split(" ") if token]
