"""Tests for FastAPI endpoints."""

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health_endpoint():
    resp = client.get("/api/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"


def test_healthcheck_endpoint():
    resp = client.get("/api/healthcheck")
    assert resp.status_code == 200
    assert resp.json()["service"] == "cancerhawk"


def test_models_endpoint():
    resp = client.get("/api/models")
    assert resp.status_code == 200
    data = resp.json()
    assert "models" in data
    assert "defaults" in data
    assert isinstance(data["models"], list)
    assert isinstance(data["defaults"], dict)


def test_root_returns_html():
    resp = client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "CancerHawk" in resp.text
