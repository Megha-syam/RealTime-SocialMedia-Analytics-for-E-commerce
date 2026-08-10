from app import create_app
from app.api import analytics


def _build_app():
    return create_app({"TESTING": True, "SQLALCHEMY_DATABASE_URI": "sqlite:///:memory:"})


def _auth_header(client):
    register_payload = {
        "full_name": "Test User",
        "email": "instant.summary@example.com",
        "password": "StrongPass123!",
    }
    client.post("/api/v1/auth/register", json=register_payload)
    login_response = client.post(
        "/api/v1/auth/login",
        json={"email": register_payload["email"], "password": register_payload["password"]},
    )
    token = login_response.get_json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


def test_instant_summary_requires_input():
    app = _build_app()
    with app.test_client() as client:
        headers = _auth_header(client)
        response = client.post(
            "/api/v1/products/summary/instant",
            json={"product": "iPhone 16 Pro"},
            headers=headers,
        )

    assert response.status_code == 400
    assert response.get_json()["error"] == "reviews or input text is required"


def test_instant_summary_returns_summary(monkeypatch):
    app = _build_app()
    monkeypatch.setattr(
        analytics,
        "generate_instant_summary",
        lambda **_: {
            "overall": "Based on instant input, sentiment is positive.",
            "pros_paragraph": "Battery and camera are appreciated.",
            "cons_paragraph": "Heating is mentioned in a few reviews.",
            "recommendation_paragraph": "Use sentiment + trend before pricing updates.",
            "sample_size": 2,
            "window_minutes": 240,
            "signal_quality": "low",
            "ai_model": "gemini",
            "source_breakdown": {"instant_input": 2},
            "pros": [],
            "cons": [],
            "pros_keyphrases": [],
            "cons_keyphrases": [],
        },
    )

    with app.test_client() as client:
        headers = _auth_header(client)
        response = client.post(
            "/api/v1/products/summary/instant",
            json={
                "product": "iPhone 16 Pro",
                "category": "smartphone",
                "input": "Battery backup is strong\nCamera is excellent in low light",
            },
            headers=headers,
        )

    assert response.status_code == 200
    data = response.get_json()
    assert data["mode"] == "instant_input"
    assert data["product"] == "iPhone 16 Pro"
    assert data["summary"]["sample_size"] == 2
