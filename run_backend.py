"""Lightweight runner for the FastAPI backend that uses the dev requirements set (avoids heavy ML deps).
Run with: python run_backend.py
"""
import os
import uvicorn

if __name__ == "__main__":
    host = os.environ.get("BACKEND_HOST", "127.0.0.1")
    port = int(os.environ.get("BACKEND_PORT", "8000"))
    print(f"Starting backend on {host}:{port}")
    uvicorn.run("backend.main:app", host=host, port=port, reload=True)
