"""Job tracking system for CancerHawk runs.

Stores job metadata in ``jobs.json`` so every user run creates a
discoverable job card that can be inspected later.

Each job record:
  - job_id:          unique ULID (chronological, URL-safe)
  - created_at:      ISO-8601 timestamp
  - research_goal:   the goal string the user provided
  - status:          pending | running | completed | failed
  - config:          model selection, submitter count, etc.
  - result:          populated when the run finishes
  - error:           populated when the run fails
"""

from __future__ import annotations

import json
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

JOBS_FILE = Path(__file__).resolve().parent.parent / "jobs.json"
_lock = threading.Lock()


def _load_jobs() -> list[dict]:
    if not JOBS_FILE.exists():
        return []
    try:
        return json.loads(JOBS_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []


def _save_jobs(jobs: list[dict]) -> None:
    JOBS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with _lock:
        JOBS_FILE.write_text(
            json.dumps(jobs, indent=2, default=str), encoding="utf-8"
        )


def create_job(*, research_goal: str, config: dict[str, Any]) -> dict:
    """Insert a new job and return it."""
    jobs = _load_jobs()
    job = {
        "job_id": _ulid(),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "research_goal": research_goal,
        "status": "pending",
        "config": config,
        "result": None,
        "error": None,
    }
    jobs.append(job)
    _save_jobs(jobs)
    return job


def update_job_status(job_id: str, status: str, **kwargs) -> Optional[dict]:
    """Update an existing job by ``job_id`` and return it."""
    jobs = _load_jobs()
    for job in jobs:
        if job.get("job_id") == job_id:
            job["status"] = status
            for k, v in kwargs.items():
                if k in ("result", "error", "config"):
                    job[k] = v
            _save_jobs(jobs)
            return job
    return None


def get_job(job_id: str) -> Optional[dict]:
    for job in _load_jobs():
        if job.get("job_id") == job_id:
            return job
    return None


def list_jobs(limit: int = 50, status: Optional[str] = None) -> list[dict]:
    jobs = _load_jobs()
    if status:
        jobs = [j for j in jobs if j.get("status") == status]
    return jobs[-limit:][::-1]  # newest first


# ---------------------------------------------------------------------------
# ULID generation (no external deps)
# ---------------------------------------------------------------------------

_ulid_counter = 0
_ulid_lock = threading.Lock()

_BASE32 = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"


def _ulid() -> str:
    """Generate a URL-safe, chronological ULID string."""
    global _ulid_counter
    with _ulid_lock:
        now_ms = int(time.time() * 1000)
        _ulid_counter = (_ulid_counter + 1) % 0x10000
        rand = _ulid_counter
    return _encode_time(now_ms) + _encode_random(rand)


def _encode_time(ms: int) -> str:
    s = ""
    for _ in range(10):
        ms, rem = divmod(ms, 32)
        s = _BASE32[rem] + s
    return s


def _encode_random(seed: int) -> str:
    s = ""
    for _ in range(16):
        seed, rem = divmod(seed, 32)
        s = _BASE32[rem] + s
    return s
