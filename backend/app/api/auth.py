from flask import Blueprint, jsonify, request
from flask_jwt_extended import create_access_token, get_jwt_identity, jwt_required

from app.extensions import db
from app.models import User
from app.utils.security import hash_password, verify_password

auth_bp = Blueprint("auth", __name__)


@auth_bp.post("/auth/register")
def register():
    payload = request.get_json(silent=True) or {}
    email = (payload.get("email") or "").strip().lower()
    password = payload.get("password") or ""
    full_name = (payload.get("full_name") or "").strip()

    if not email or not password or not full_name:
        return jsonify({"error": "email, password and full_name are required"}), 400
    if User.query.filter_by(email=email).first():
        return jsonify({"error": "email already exists"}), 409

    user = User(email=email, password_hash=hash_password(password), full_name=full_name)
    db.session.add(user)
    db.session.commit()
    return jsonify({"message": "registered"}), 201


@auth_bp.post("/auth/login")
def login():
    payload = request.get_json(silent=True) or {}
    email = (payload.get("email") or "").strip().lower()
    password = payload.get("password") or ""

    user = User.query.filter_by(email=email).first()
    if not user or not verify_password(user.password_hash, password):
        return jsonify({"error": "invalid credentials"}), 401

    token = create_access_token(
        identity=str(user.id),
        additional_claims={"email": user.email, "role": user.role},
    )
    return jsonify(
        {
            "access_token": token,
            "user": {
                "id": user.id,
                "email": user.email,
                "full_name": user.full_name,
                "role": user.role,
            },
        }
    ), 200


@auth_bp.get("/auth/profile")
@jwt_required()
def profile():
    user_id = get_jwt_identity()
    user = User.query.get(int(user_id))
    if not user:
        return jsonify({"error": "user not found"}), 404
    return jsonify(
        {
            "id": user.id,
            "email": user.email,
            "full_name": user.full_name,
            "role": user.role,
        }
    ), 200


@auth_bp.put("/auth/profile")
@jwt_required()
def update_profile():
    user_id = get_jwt_identity()
    user = User.query.get(int(user_id))
    if not user:
        return jsonify({"error": "user not found"}), 404

    payload = request.get_json(silent=True) or {}
    full_name = (payload.get("full_name") or "").strip()
    if not full_name:
        return jsonify({"error": "full_name is required"}), 400

    user.full_name = full_name
    db.session.commit()
    return jsonify(
        {
            "id": user.id,
            "email": user.email,
            "full_name": user.full_name,
            "role": user.role,
        }
    ), 200


@auth_bp.post("/auth/change-password")
@jwt_required()
def change_password():
    user_id = get_jwt_identity()
    user = User.query.get(int(user_id))
    if not user:
        return jsonify({"error": "user not found"}), 404

    payload = request.get_json(silent=True) or {}
    current_password = payload.get("current_password") or ""
    new_password = payload.get("new_password") or ""

    if not current_password or not new_password:
        return jsonify({"error": "current_password and new_password are required"}), 400
    if not verify_password(user.password_hash, current_password):
        return jsonify({"error": "current password is incorrect"}), 400
    if len(new_password) < 8:
        return jsonify({"error": "new_password must be at least 8 characters"}), 400
    if verify_password(user.password_hash, new_password):
        return jsonify({"error": "new password must be different from current password"}), 400

    user.password_hash = hash_password(new_password)
    db.session.commit()
    return jsonify({"message": "password updated"}), 200
