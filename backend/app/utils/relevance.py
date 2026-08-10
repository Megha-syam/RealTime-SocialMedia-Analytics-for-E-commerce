from app.utils.text import clean_text

OFF_TOPIC_TERMS = {
    "emulator",
    "azahar",
    "turnip",
    "rom",
    "bios",
    "sideload",
    "mod apk",
    "jailbreak",
    "gpu driver",
    "style savvy",
}

COMMERCE_TERMS = {
    "buy",
    "price",
    "cost",
    "sale",
    "discount",
    "deal",
    "offer",
    "warranty",
    "service",
    "delivery",
    "shipping",
    "return",
    "refund",
    "review",
    "rating",
    "camera",
    "battery",
    "performance",
    "heating",
    "quality",
    "complaint",
    "issue",
    "availability",
}

CONTEXT_HINTS = {
    "tv": {"tv", "television", "oled", "qled", "led"},
    "bike": {"bike", "motorcycle", "scooter", "enfield", "bullet", "himalayan", "pulsar"},
    "motorcycle": {"bike", "motorcycle", "scooter", "enfield", "bullet", "himalayan"},
    "scooter": {"bike", "motorcycle", "scooter"},
    "car": {"car", "vehicle", "suv", "sedan", "ev"},
    "suv": {"car", "vehicle", "suv"},
    "sedan": {"car", "vehicle", "sedan"},
    "ev": {"ev", "electric", "vehicle", "car"},
    "laptop": {"laptop", "notebook", "macbook", "thinkpad", "vivobook"},
    "notebook": {"laptop", "notebook"},
    "tablet": {"tablet", "ipad", "tab"},
    "watch": {"watch", "smartwatch", "wearable", "band"},
    "smartwatch": {"watch", "smartwatch", "wearable", "band"},
    "earbuds": {"earbuds", "buds", "airpods", "headphones"},
    "buds": {"earbuds", "buds", "airpods", "headphones"},
    "fridge": {"fridge", "refrigerator"},
    "refrigerator": {"fridge", "refrigerator"},
    "microwave": {"microwave", "oven"},
    "ac": {"ac", "air conditioner", "aircon"},
}


def _contains_term(text: str, term: str) -> bool:
    return f" {term} " in f" {text} "


def _context_groups_for_query(normalized_product: str):
    tokens = [token for token in normalized_product.split() if token]
    groups = []
    for token in tokens:
        aliases = CONTEXT_HINTS.get(token)
        if aliases:
            groups.append(aliases)
    return groups


def is_off_topic(text: str) -> bool:
    normalized = clean_text(text)
    if not normalized:
        return True
    return any(term in normalized for term in OFF_TOPIC_TERMS)


def is_product_mention(text: str, product: str) -> bool:
    normalized_text = clean_text(text)
    normalized_product = clean_text(product)
    product_tokens = [token for token in normalized_product.split() if token]
    if not normalized_text or not product_tokens:
        return False

    # Strong match: full phrase.
    if normalized_product in normalized_text:
        return True

    # For specific product types (e.g., TV, bike), require type evidence in text.
    context_groups = _context_groups_for_query(normalized_product)
    for aliases in context_groups:
        if not any(_contains_term(normalized_text, alias) for alias in aliases):
            return False

    alpha_tokens = [token for token in product_tokens if token.isalpha() and len(token) >= 3]
    numeric_tokens = [token for token in product_tokens if token.isdigit()]

    if alpha_tokens and not all(token in normalized_text for token in alpha_tokens):
        return False

    if alpha_tokens and numeric_tokens:
        for alpha in alpha_tokens:
            for number in numeric_tokens:
                if f"{alpha} {number}" in normalized_text or f"{alpha}{number}" in normalized_text:
                    return True
        return False

    if numeric_tokens and not any(token in normalized_text for token in numeric_tokens):
        return False

    if alpha_tokens:
        return True
    return any(token in normalized_text for token in product_tokens)


def is_commerce_context(text: str) -> bool:
    normalized = clean_text(text)
    if not normalized or is_off_topic(normalized):
        return False
    return any(term in normalized for term in COMMERCE_TERMS)
