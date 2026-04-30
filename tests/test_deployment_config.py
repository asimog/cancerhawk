"""Deployment configuration regression tests."""

from app import main


def test_railway_worker_binds_to_public_interface_by_default():
    assert main.HOST == "0.0.0.0"


def test_frontend_backend_url_has_no_dead_railway_fallback():
    source = open("src/lib/blocks.ts", encoding="utf-8").read()

    assert "cancerhawk-production.up.railway.app" not in source
    assert ".trim().replace" in source
