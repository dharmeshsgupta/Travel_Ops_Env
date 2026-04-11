"""ASGI entry for Hugging Face / deployment (`uvicorn app:app`)."""
from server.app import app

__all__ = ["app"]
