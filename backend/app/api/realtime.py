from flask import request
from flask_socketio import emit

from app.extensions import socketio


@socketio.on("connect", namespace="/stream")
def on_connect():
    emit("connected", {"message": "Realtime stream connected", "sid": request.sid})


@socketio.on("subscribe_product", namespace="/stream")
def on_subscribe(payload):
    product = (payload or {}).get("product", "all")
    emit("subscription_ack", {"product": product, "status": "subscribed"})
