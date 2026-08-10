import atexit
import logging

from flask import Flask, jsonify
from dotenv import load_dotenv
import structlog

from app.api.analytics import analytics_bp
from app.api.auth import auth_bp
from app.api.health import health_bp
from app.api import realtime  # noqa: F401
from app.config import Config
from app.extensions import cors, db, jwt, metrics, socketio
from app.services.scheduler import start_scheduler, stop_scheduler


def create_app(config_override: dict | None = None) -> Flask:
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.add_log_level,
            structlog.processors.JSONRenderer(),
        ]
    )

    app = Flask(__name__)
    app.config.from_object(Config)
    if config_override:
        app.config.update(config_override)

    db.init_app(app)
    jwt.init_app(app)
    cors.init_app(app, resources={r"/api/*": {"origins": app.config["CORS_ORIGINS"]}})
    socketio.init_app(app)

    if app.config.get("ENABLE_PROMETHEUS", True):
        metrics.init_app(app)

    app.register_blueprint(health_bp, url_prefix="/api/v1")
    app.register_blueprint(auth_bp, url_prefix="/api/v1")
    app.register_blueprint(analytics_bp, url_prefix="/api/v1")

    with app.app_context():
        db.create_all()

    if not app.config.get("TESTING", False):
        start_scheduler(app)
        atexit.register(stop_scheduler)

    @app.errorhandler(404)
    def not_found(_):
        return jsonify({"error": "not found"}), 404

    @app.errorhandler(500)
    def server_error(_):
        return jsonify({"error": "internal server error"}), 500

    return app
