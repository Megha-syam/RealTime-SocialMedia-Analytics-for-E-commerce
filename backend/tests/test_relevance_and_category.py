from app.services import category_service
from app.utils.relevance import is_product_mention


def test_category_uses_heuristic_when_gemini_returns_other(monkeypatch):
    monkeypatch.setattr(category_service, "classify_with_gemini", lambda _: "other")
    assert category_service.classify_product_category("Samsung TV") == "tv"


def test_category_detects_royal_enfield_as_bike(monkeypatch):
    monkeypatch.setattr(category_service, "classify_with_gemini", lambda _: "")
    assert category_service.classify_product_category("Royal Enfield Hunter 350") == "bike"


def test_product_mention_rejects_phone_post_for_samsung_tv():
    text = "Samsung Galaxy phone battery review and camera comparison"
    assert not is_product_mention(text, "Samsung TV")


def test_product_mention_accepts_actual_samsung_tv_post():
    text = "Samsung TV QLED display quality and sound review is excellent"
    assert is_product_mention(text, "Samsung TV")
