from app.services import summary_service


def test_generate_instant_summary_from_multiline_input(monkeypatch):
    monkeypatch.setattr(
        summary_service,
        "score_sentiment",
        lambda _: {"label": "positive", "score": 0.6},
    )
    monkeypatch.setattr(summary_service, "enhance_summary", lambda **_: None)

    result = summary_service.generate_instant_summary(
        product_name="iPhone 16 Pro",
        reviews_input="Battery life is excellent.\nDisplay is bright and smooth.",
        window_minutes=60,
    )

    assert result["sample_size"] == 2
    assert result["window_minutes"] == 60
    assert result["ai_model"] == "deterministic"
    assert "Based on 2 live mentions" in result["overall"]


def test_generate_instant_summary_from_structured_reviews(monkeypatch):
    monkeypatch.setattr(summary_service, "enhance_summary", lambda **_: None)

    result = summary_service.generate_instant_summary(
        product_name="iPhone 16 Pro",
        reviews_input=[
            {"review": "Camera is amazing.", "sentiment": "positive", "rating": 5},
            {"text": "Phone gets warm during gaming.", "sentiment_label": "negative"},
        ],
        window_minutes=120,
    )

    assert result["sample_size"] == 2
    assert result["source_breakdown"]["instant_input"] == 2
