"""Regression and end-to-end tests for the job card system.

Covers:
  - Job creation, status updates, listing, and retrieval
  - Job API endpoints (mocked)
  - Integration with WebSocket run handler
  - Existing paper_engine / openrouter fixes (regression)
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Ensure app/ is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.jobs import (
    _BASE32,
    _ulid,
    append_job_event,
    create_job,
    find_job_by_idempotency_key,
    get_job,
    get_jobs_file,
    job_store_info,
    list_jobs,
    update_job_status,
)
from app.openrouter import _extract_json


# ---------------------------------------------------------------------------
# Job storage regression / unit tests
# ---------------------------------------------------------------------------

class TestJobStorage:
    """Test the job storage layer in isolation."""

    def setup_method(self):
        # Use a temporary jobs file for each test
        self.test_file = Path(os.path.join(os.path.dirname(__file__), "..", "test_jobs.json"))
        if self.test_file.exists():
            self.test_file.unlink()
        self.env_patches = [
            patch.dict(os.environ, {}, clear=False),
            patch.dict(os.environ, {
                "CANCERHAWK_JOBS_FILE": "",
                "CANCERHAWK_JOBS_PATH": "",
                "RAILWAY_VOLUME_MOUNT_PATH": "",
            }, clear=False),
        ]
        for env_patch in self.env_patches:
            env_patch.start()
        # Patch JOBS_FILE to point to our test file (as a Path object)
        self.patcher = patch("app.jobs.JOBS_FILE", self.test_file)
        self.patcher.start()

    def teardown_method(self):
        self.patcher.stop()
        for env_patch in reversed(self.env_patches):
            env_patch.stop()
        if os.path.exists(self.test_file):
            os.remove(self.test_file)

    def test_create_job(self):
        job = create_job(research_goal="cure cancer", config={"models": "test"})
        assert job["research_goal"] == "cure cancer"
        assert job["status"] == "pending"
        assert job["job_id"]
        assert job["created_at"]
        assert job["result"] is None
        assert job["error"] is None

    def test_create_job_has_unique_ids(self):
        j1 = create_job(research_goal="goal1", config={})
        j2 = create_job(research_goal="goal2", config={})
        assert j1["job_id"] != j2["job_id"]

    def test_get_job_exists(self):
        job = create_job(research_goal="test", config={})
        retrieved = get_job(job["job_id"])
        assert retrieved is not None
        assert retrieved["job_id"] == job["job_id"]

    def test_get_job_not_exists(self):
        result = get_job("nonexistent-id")
        assert result is None

    def test_update_job_status(self):
        job = create_job(research_goal="test", config={})
        updated = update_job_status(job["job_id"], "running")
        assert updated is not None
        assert updated["status"] == "running"

        # Verify persistence
        retrieved = get_job(job["job_id"])
        assert retrieved["status"] == "running"
        assert retrieved["updated_at"]

    def test_append_job_event(self):
        job = create_job(research_goal="test", config={})
        append_job_event(job["job_id"], stage="paper", message="paper started", data={"x": 1})
        retrieved = get_job(job["job_id"])
        assert retrieved["events"][0]["stage"] == "paper"
        assert retrieved["events"][0]["message"] == "paper started"
        assert retrieved["events"][0]["data"] == {"x": 1}

    def test_update_job_result(self):
        job = create_job(research_goal="test", config={})
        result_data = {"title": "Test Paper", "block": 1}
        update_job_status(job["job_id"], "completed", result=result_data)
        retrieved = get_job(job["job_id"])
        assert retrieved["status"] == "completed"
        assert retrieved["result"]["title"] == "Test Paper"

    def test_find_job_by_idempotency_key(self):
        job = create_job(research_goal="test", config={"idempotency_key": "abc123"})
        assert find_job_by_idempotency_key("abc123")["job_id"] == job["job_id"]
        assert find_job_by_idempotency_key("missing") is None

    def test_update_job_error(self):
        job = create_job(research_goal="test", config={})
        update_job_status(job["job_id"], "failed", error="something broke")
        retrieved = get_job(job["job_id"])
        assert retrieved["status"] == "failed"
        assert "something broke" in retrieved["error"]

    def test_list_jobs_newest_first(self):
        j1 = create_job(research_goal="first", config={})
        time.sleep(0.01)
        j2 = create_job(research_goal="second", config={})
        jobs = list_jobs(limit=10)
        assert len(jobs) == 2
        assert jobs[0]["job_id"] == j2["job_id"]  # newest first

    def test_list_jobs_limit(self):
        for i in range(5):
            create_job(research_goal=f"goal{i}", config={})
        jobs = list_jobs(limit=3)
        assert len(jobs) == 3

    def test_list_jobs_by_status(self):
        j1 = create_job(research_goal="pending job", config={})
        j2 = create_job(research_goal="completed job", config={})
        update_job_status(j2["job_id"], "completed")

        pending = list_jobs(status="pending")
        assert all(j["status"] == "pending" for j in pending)

        completed = list_jobs(status="completed")
        assert all(j["status"] == "completed" for j in completed)

    def test_job_store_uses_explicit_durable_file(self, tmp_path):
        jobs_file = tmp_path / "durable" / "jobs.json"
        with patch.dict(os.environ, {"CANCERHAWK_JOBS_FILE": str(jobs_file)}, clear=False):
            job = create_job(research_goal="durable", config={})
            assert get_jobs_file() == jobs_file
            assert jobs_file.exists()
            assert get_job(job["job_id"])["research_goal"] == "durable"
            assert job_store_info()["durable"] is True

    def test_job_store_uses_railway_volume_mount(self, tmp_path):
        with patch.dict(os.environ, {"RAILWAY_VOLUME_MOUNT_PATH": str(tmp_path)}, clear=False):
            expected = tmp_path / "cancerhawk" / "jobs.json"
            assert get_jobs_file() == expected
            assert job_store_info()["durable"] is True

    def test_job_store_warns_on_railway_without_durable_path(self):
        with patch.dict(os.environ, {
            "CANCERHAWK_JOBS_FILE": "",
            "CANCERHAWK_JOBS_PATH": "",
            "RAILWAY_VOLUME_MOUNT_PATH": "",
            "RAILWAY_ENVIRONMENT_NAME": "production",
        }, clear=False):
            info = job_store_info()
            assert info["durable"] is False
            assert "attach a Railway volume" in info["warning"]


class TestULIDGeneration:
    """Test ULID generation."""

    def test_ulid_format(self):
        uid = _ulid()
        assert len(uid) == 26  # 10 time chars + 16 random chars
        assert all(c in _BASE32 for c in uid)

    def test_ulid_unique(self):
        ids = {_ulid() for _ in range(100)}
        assert len(ids) == 100  # all unique


# ---------------------------------------------------------------------------
# openrouter._extract_json regression tests (the list-wrapping fix)
# ---------------------------------------------------------------------------

class TestExtractJsonRegression:
    """Ensure the array-wrapping fix doesn't regress."""

    def test_dict_normal(self):
        result = _extract_json('{"accept": false, "scores": {"novelty": 5}}')
        assert isinstance(result, dict)
        assert result["accept"] is False

    def test_array_wrapped_single_dict(self):
        result = _extract_json('[{"accept": false, "scores": {"novelty": 5}}]')
        assert isinstance(result, dict)
        assert result["accept"] is False

    def test_array_multiple_elements_raises(self):
        from app.openrouter import OpenRouterError
        try:
            _extract_json('[{"a": 1}, {"b": 2}]')
            assert False, "should have raised"
        except OpenRouterError:
            pass

    def test_array_empty_raises(self):
        from app.openrouter import OpenRouterError
        try:
            _extract_json('[]')
            assert False, "should have raised"
        except OpenRouterError:
            pass


# ---------------------------------------------------------------------------
# API endpoint tests (calling functions directly)
# ---------------------------------------------------------------------------

class TestJobAPIEndpoints:
    """Test the job API endpoints by calling the functions directly."""

    def setup_method(self):
        # Use a temporary jobs file for each test
        self.test_file = Path(os.path.join(os.path.dirname(__file__), "..", "test_jobs_api.json"))
        if self.test_file.exists():
            self.test_file.unlink()
        self.env_patch = patch.dict(os.environ, {
            "CANCERHAWK_JOBS_FILE": "",
            "CANCERHAWK_JOBS_PATH": "",
            "RAILWAY_VOLUME_MOUNT_PATH": "",
        }, clear=False)
        self.env_patch.start()
        # Patch JOBS_FILE to point to our test file
        self.patcher = patch("app.jobs.JOBS_FILE", self.test_file)
        self.patcher.start()

    def teardown_method(self):
        self.patcher.stop()
        self.env_patch.stop()
        if self.test_file.exists():
            self.test_file.unlink()

    def test_list_jobs_empty(self):
        from app.main import list_jobs as list_jobs_endpoint
        jobs = list_jobs_endpoint(limit=50, status=None)
        assert jobs == []

    def test_list_jobs_after_creation(self):
        job = create_job(research_goal="test goal", config={"x": 1})
        from app.main import list_jobs as list_jobs_endpoint
        jobs = list_jobs_endpoint(limit=50, status=None)
        assert len(jobs) == 1
        assert jobs[0]["research_goal"] == "test goal"

    def test_list_jobs_with_status_filter(self):
        j1 = create_job(research_goal="pending", config={})
        j2 = create_job(research_goal="completed", config={})
        update_job_status(j2["job_id"], "completed")

        from app.main import list_jobs as list_jobs_endpoint
        pending = list_jobs_endpoint(limit=50, status="pending")
        assert all(j["status"] == "pending" for j in pending)

        completed = list_jobs_endpoint(limit=50, status="completed")
        assert all(j["status"] == "completed" for j in completed)

    def test_get_job_details(self):
        job = create_job(research_goal="detail test", config={})
        update_job_status(job["job_id"], "completed", result={"title": "Test"})

        from app.main import get_job as get_job_endpoint
        data = get_job_endpoint(job["job_id"])
        assert data is not None
        assert data["research_goal"] == "detail test"
        assert data["status"] == "completed"
        assert data["result"]["title"] == "Test"

    def test_get_job_not_found(self):
        from app.main import get_job as get_job_endpoint
        data = get_job_endpoint("nonexistent-id")
        assert data is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
