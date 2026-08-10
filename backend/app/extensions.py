from flask_cors import CORS
from flask_jwt_extended import JWTManager
from flask_socketio import SocketIO
from flask_sqlalchemy import SQLAlchemy
from prometheus_flask_exporter import PrometheusMetrics

db = SQLAlchemy()
jwt = JWTManager()
cors = CORS()
socketio = SocketIO(async_mode="eventlet", cors_allowed_origins="*")
metrics = PrometheusMetrics.for_app_factory()
