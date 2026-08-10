from app import create_app
from app.api import analytics


def _build_app():
    return create_app({"TESTING": True, "SQLALCHEMY_DATABASE_URI": "sqlite:///:memory:"})


def _auth_header(client):
    register_payload = {
        "full_name": "Test User",
        "email": "test.user@example.com",
        "password": "StrongPass123!",
    }
    client.post("/api/v1/auth/register", json=register_payload)
    login_response = client.post(
        "/api/v1/auth/login",
        json={"email": register_payload["email"], "password": register_payload["password"]},
    )
    token = login_response.get_json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


def test_original_reviews_endpoint_rejects_invalid_category():
    app = _build_app()
    with app.test_client() as client:
        headers = _auth_header(client)
        response = client.post(
            "/api/v1/products/reviews/original",
            json={"product": "iPhone 16 Pro", "category": "furniture"},
            headers=headers,
        )

    assert response.status_code == 400
    data = response.get_json()
    assert data["error"] == "invalid category"
    assert "smartphone" in data["allowed_categories"]


def test_original_reviews_endpoint_returns_gemini_reviews(monkeypatch):
    app = _build_app()

    monkeypatch.setattr(
        analytics,
        "generate_original_reviews",
        lambda **_: {
            "product": "iPhone 16 Pro",
            "category": "smartphone",
            "source": "gemini",
            "reviews": [
                {"sentiment": "positive", "rating": 5, "review": "Battery life is excellent."},
                {"sentiment": "neutral", "rating": 3, "review": "Display is good, speaker is average."},
            ],
        },
    )

    with app.test_client() as client:
        headers = _auth_header(client)
        response = client.post(
            "/api/v1/products/reviews/original",
            json={"product": "iPhone 16 Pro", "category": "smart phone", "count": 2},
            headers=headers,
        )

    assert response.status_code == 200
    data = response.get_json()
    assert data["source"] == "gemini"
    assert data["category"] == "smartphone"
    assert len(data["reviews"]) == 2
