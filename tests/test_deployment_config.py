"""Deployment configuration regression tests."""

from app import main


def test_railway_worker_binds_to_public_interface_by_default():
    assert main.HOST == "0.0.0.0"


def test_frontend_backend_url_has_no_dead_railway_fallback():
    source = open("src/lib/blocks.ts", encoding="utf-8").read()

    assert "cancerhawk-production.up.railway.app" not in source
    assert ".trim().replace" in source


def test_railway_build_installs_worker_requirements_only():
    source = open("nixpacks.toml", encoding="utf-8").read()

    assert "-r app/requirements.txt" in source
    assert "-r requirements.txt" not in source


def test_railway_source_context_excludes_frontend_build_state():
    source = open(".railwayignore", encoding="utf-8").read()

    assert "package.json" in source
    assert "package-lock.json" in source
    assert "tsconfig.tsbuildinfo" in source
    assert "pages" in source
    assert "src" in source
    assert "public" in source


def test_autonomous_generation_is_opt_in():
    source = open("app/main.py", encoding="utf-8").read()

    assert "HERMES_AUTO_GENERATE_ENABLED" in source
    assert "auto_generation_skipped_disabled" in source
