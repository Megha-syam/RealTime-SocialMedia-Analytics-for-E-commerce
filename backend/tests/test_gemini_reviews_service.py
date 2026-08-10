from app.services import gemini_service


def test_generate_original_reviews_normalizes_and_dedupes(monkeypatch):
    monkeypatch.setattr(
        gemini_service,
        "_json_request",
        lambda *args, **kwargs: {
            "product": "iPhone 16 Pro",
            "category": "smartphone",
            "reviews": [
                {"sentiment": "positive", "rating": 5, "review": "Great camera and battery life."},
                {"sentiment": "positive", "rating": 5, "review": "Great camera and battery life."},
                {"sentiment": "mixed", "rating": 7, "review": "Fast performance but average thermals."},
            ],
        },
    )

    result = gemini_service.generate_original_reviews(
        product_name="iPhone 16 Pro",
        category="smart phone",
        count=3,
    )

    assert result is not None
    assert result["category"] == "smartphone"
    assert result["source"] == "gemini"
    assert len(result["reviews"]) == 2
    assert result["reviews"][1]["sentiment"] == "neutral"
    assert result["reviews"][1]["rating"] == 5


def test_generate_original_reviews_rejects_invalid_category():
    result = gemini_service.generate_original_reviews(
        product_name="iPhone 16 Pro",
        category="furniture",
        count=3,
    )
    assert result is None
