import asyncio
import os
from pathlib import Path
from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def isolate_job_store(tmp_path, monkeypatch):
    test_jobs_file = tmp_path / "jobs.json"
    test_history_file = tmp_path / "jobs-history.json"

    monkeypatch.delenv("CANCERHAWK_JOBS_FILE", raising=False)
    monkeypatch.delenv("CANCERHAWK_JOBS_PATH", raising=False)
    monkeypatch.delenv("RAILWAY_VOLUME_MOUNT_PATH", raising=False)
    monkeypatch.delenv("RAILWAY_ENVIRONMENT", raising=False)
    monkeypatch.delenv("RAILWAY_PROJECT_ID", raising=False)
    monkeypatch.delenv("HERMES_AUTO_GENERATE_ENABLED", raising=False)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("GITHUB_REPO", raising=False)
    monkeypatch.setenv("CANCERHAWK_JOBS_FILE", str(test_jobs_file))

    from app import jobs

    monkeypatch.setattr(jobs, "JOBS_FILE", test_jobs_file)
    monkeypatch.setattr(jobs, "JOBS_HISTORY_FILE", test_history_file)


@pytest.fixture(autouse=True)
def suppress_background_workers(monkeypatch):
    async def noop_worker():
        while True:
            await asyncio.sleep(86400)

    monkeypatch.setattr("app.main.publish_cycle_worker", noop_worker)
    monkeypatch.setattr("app.main.autonomous_generation_worker", noop_worker)
    monkeypatch.setattr("app.main.daily_git_sync_worker", noop_worker)
