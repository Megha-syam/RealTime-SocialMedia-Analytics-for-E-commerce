from app.services.gemini_service import classify_product_category as classify_with_gemini
from app.utils.text import clean_text


def _heuristic_category(product_name: str) -> str:
    normalized = clean_text(product_name)
    if not normalized:
        return "other"

    strong_phrases = {
        "tv": [" smart tv", " oled tv", " qled tv", " television", " samsung tv", " bravia"],
        "bike": ["royal enfield", "classic 350", "hunter 350", "meteor 350", "himalayan"],
        "wearable": ["smart watch", "smartwatch", "fitness band", "ear buds"],
        "appliance": ["washing machine", "air conditioner", "dish washer"],
    }
    padded = f" {normalized} "
    for category, phrases in strong_phrases.items():
        if any(phrase in padded for phrase in phrases):
            return category

    mapping = {
        "smartphone": [
            "iphone",
            "oneplus",
            "pixel",
            "redmi",
            "realme",
            "galaxy",
            "phone",
            "android",
        ],
        "laptop": ["laptop", "macbook", "notebook", "thinkpad", "vivobook", "ultrabook"],
        "tablet": ["ipad", "tablet", "tab "],
        "tv": ["tv", "television", "oled", "qled", "bravia"],
        "wearable": ["watch", "smartwatch", "band", "earbuds", "airpods", "buds"],
        "automotive": ["car", "ev", "vehicle", "tesla", "suv", "sedan"],
        "bike": [
            "bike",
            "motorcycle",
            "bajaj",
            "ct ",
            "pulsar",
            "scooter",
            "enfield",
            "royal enfield",
            "bullet",
            "himalayan",
            "hunter",
            "classic 350",
            "meteor",
        ],
        "appliance": ["fridge", "refrigerator", "ac", "washing machine", "microwave"],
        "gaming": ["ps5", "xbox", "nintendo", "game", "gpu", "console"],
    }
    for category, keywords in mapping.items():
        if any(keyword in normalized for keyword in keywords):
            return category
    return "other"


def classify_product_category(product_name: str) -> str:
    base = _heuristic_category(product_name)
    gemini = classify_with_gemini(product_name)
    if gemini and gemini != "other":
        return gemini
    if base != "other":
        return base
    return gemini or "other"
