from app import create_app
from app.extensions import db
from app.models import Product
from app.services import category_service, gemini_service, ingestion_service


def _build_app():
    return create_app({"TESTING": True, "SQLALCHEMY_DATABASE_URI": "sqlite:///:memory:"})


def test_category_service_prefers_gemini(monkeypatch):
    monkeypatch.setattr(category_service, "classify_with_gemini", lambda _: "tablet")

    assert category_service.classify_product_category("iPhone 16 Pro") == "tablet"


def test_gemini_category_normalizes_alias(monkeypatch):
    monkeypatch.setattr(
        gemini_service,
        "_json_request",
        lambda *args, **kwargs: {"category": "Smart Phone"},
    )

    assert gemini_service.classify_product_category("iPhone 16 Pro") == "smartphone"


def test_get_or_create_product_updates_other_category(monkeypatch):
    app = _build_app()
    with app.app_context():
        db.session.add(
            Product(slug="iphone-16-pro", display_name="iPhone 16 Pro", category="other")
        )
        db.session.commit()

        monkeypatch.setattr(
            ingestion_service, "classify_product_category", lambda _: "smartphone"
        )
        product = ingestion_service.get_or_create_product("iPhone 16 Pro")

        assert product.category == "smartphone"
