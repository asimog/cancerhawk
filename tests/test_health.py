"""Tests for health check endpoints."""

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_returns_200():
    resp = client.get("/api/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "service" in data


def test_healthcheck_returns_200():
    resp = client.get("/api/healthcheck")
    assert resp.status_code == 200


def test_health_has_json_content_type():
    resp = client.get("/api/health")
    assert "application/json" in resp.headers.get("content-type", "")


def test_healthcheck_has_json_content_type():
    resp = client.get("/api/healthcheck")
    assert "application/json" in resp.headers.get("content-type", "")
