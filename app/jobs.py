"""Job tracking system for CancerHawk runs.

Stores job metadata in a JSON file so every user run creates a discoverable
job card that can be inspected later. In production this file can live on a
Railway volume via ``CANCERHAWK_JOBS_FILE`` or ``RAILWAY_VOLUME_MOUNT_PATH``;
local development keeps using ``jobs.json`` at the repo root.

Each job record:
  - job_id:          unique ULID (chronological, URL-safe)
  - created_at:      ISO-8601 timestamp
  - research_goal:   the goal string the user provided
  - status:          pending | running | completed | published | failed
  - config:          model selection, submitter count, etc.
  - result:          populated when the run finishes
  - error:           populated when the run fails
"""

from __future__ import annotations

import json
import os
import threading
import time
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

JOBS_FILE = Path(__file__).resolve().parent.parent / "jobs.json"
_lock = threading.RLock()
MAX_JOB_EVENTS = 300
_DURABLE_ENV_NAMES = ("CANCERHAWK_JOBS_FILE", "CANCERHAWK_JOBS_PATH")


def get_jobs_file() -> Path:
    """Return the active job store path.

    Railway containers have ephemeral application filesystems. If a volume is
    mounted, Railway exposes it through ``RAILWAY_VOLUME_MOUNT_PATH``; we store
    job state there automatically. Operators can override the path explicitly
    with ``CANCERHAWK_JOBS_FILE``.
    """
    for name in _DURABLE_ENV_NAMES:
        value = os.environ.get(name, "").strip()
        if value:
            return Path(value).expanduser()

    volume_mount = os.environ.get("RAILWAY_VOLUME_MOUNT_PATH", "").strip()
    if volume_mount:
        return Path(volume_mount).expanduser() / "cancerhawk" / "jobs.json"

    return JOBS_FILE


def job_store_info() -> dict[str, Any]:
    path = get_jobs_file()
    explicit_path = any(os.environ.get(name, "").strip() for name in _DURABLE_ENV_NAMES)
    volume_path = bool(os.environ.get("RAILWAY_VOLUME_MOUNT_PATH", "").strip())
    railway_runtime = bool(
        os.environ.get("RAILWAY_ENVIRONMENT")
        or os.environ.get("RAILWAY_ENVIRONMENT_NAME")
        or os.environ.get("RAILWAY_SERVICE_NAME")
    )
    durable = explicit_path or volume_path
    return {
        "backend": "json-file",
        "path": str(path),
        "durable": durable,
        "railway_runtime": railway_runtime,
        "warning": (
            None
            if durable or not railway_runtime
            else "Job store is on the container filesystem; attach a Railway volume or set CANCERHAWK_JOBS_FILE."
        ),
    }


def _load_jobs() -> list[dict]:
    jobs_file = get_jobs_file()
    with _lock:
        if not jobs_file.exists():
            return []
        try:
            loaded = json.loads(jobs_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return []
        return loaded if isinstance(loaded, list) else []


def _save_jobs(jobs: list[dict]) -> None:
    jobs_file = get_jobs_file()
    jobs_file.parent.mkdir(parents=True, exist_ok=True)
    with _lock:
        payload = json.dumps(jobs, indent=2, default=str)
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=str(jobs_file.parent),
            delete=False,
        ) as tmp:
            tmp.write(payload)
            tmp.flush()
            os.fsync(tmp.fileno())
            tmp_name = tmp.name
        os.replace(tmp_name, jobs_file)


def create_job(*, research_goal: str, config: dict[str, Any]) -> dict:
    """Insert a new job and return it."""
    now = datetime.now(timezone.utc).isoformat()
    with _lock:
        jobs = _load_jobs()
        job = {
            "job_id": _ulid(),
            "created_at": now,
            "updated_at": now,
            "research_goal": research_goal,
            "status": "pending",
            "config": config,
            "result": None,
            "error": None,
            "events": [],
        }
        jobs.append(job)
        _save_jobs(jobs)
        return job


def find_job_by_idempotency_key(idempotency_key: str) -> Optional[dict]:
    """Return the most recent job created with the same idempotency key."""
    key = idempotency_key.strip()
    if not key:
        return None
    for job in reversed(_load_jobs()):
        config = job.get("config") or {}
        if config.get("idempotency_key") == key:
            return job
    return None


def update_job_status(job_id: str, status: str, **kwargs) -> Optional[dict]:
    """Update an existing job by ``job_id`` and return it."""
    with _lock:
        jobs = _load_jobs()
        for job in jobs:
            if job.get("job_id") == job_id:
                job["status"] = status
                job["updated_at"] = datetime.now(timezone.utc).isoformat()
                for k, v in kwargs.items():
                    if k in ("result", "error", "config"):
                        job[k] = v
                _save_jobs(jobs)
                return job
    return None


def append_job_event(
    job_id: str,
    *,
    stage: str,
    message: str,
    data: Optional[dict[str, Any]] = None,
) -> Optional[dict]:
    """Append a live event to a job card, capping stored history."""
    with _lock:
        jobs = _load_jobs()
        for job in jobs:
            if job.get("job_id") == job_id:
                events = list(job.get("events") or [])
                events.append({
                    "at": datetime.now(timezone.utc).isoformat(),
                    "stage": stage,
                    "message": message,
                    "data": data,
                })
                job["events"] = events[-MAX_JOB_EVENTS:]
                job["updated_at"] = datetime.now(timezone.utc).isoformat()
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
