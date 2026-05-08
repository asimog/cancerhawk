"""Postgres database layer for CancerHawk.

Requires ``DATABASE_URL`` in the environment (Railway auto-links Postgres).
Uses asyncpg for async, connection-pooled access.

Tables
- jobs    : all research run job cards + events
- blocks  : published blocks (stored between daily GitHub syncs)
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Optional

import asyncpg

logger = logging.getLogger("cancerhawk.db")

_pool: asyncpg.Pool | None = None

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS jobs (
    job_id         TEXT PRIMARY KEY,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    research_goal  TEXT NOT NULL,
    status         TEXT NOT NULL DEFAULT 'pending',
    config         JSONB NOT NULL DEFAULT '{}'::jsonb,
    result         JSONB,
    error          TEXT,
    idempotency_key TEXT,
    events         JSONB NOT NULL DEFAULT '[]'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_jobs_status     ON jobs (status);
CREATE INDEX IF NOT EXISTS idx_jobs_created    ON jobs (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_jobs_idempotent ON jobs (idempotency_key) WHERE idempotency_key IS NOT NULL;

CREATE TABLE IF NOT EXISTS blocks (
    block_number   INT PRIMARY KEY,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    research_goal  TEXT NOT NULL,
    paper_title    TEXT NOT NULL DEFAULT '',
    paper_md       TEXT NOT NULL DEFAULT '',
    paper_html     TEXT NOT NULL DEFAULT '',
    analysis_json  JSONB NOT NULL DEFAULT '{}'::jsonb,
    market_price   REAL NOT NULL DEFAULT 0,
    simulation_html TEXT NOT NULL DEFAULT '',
    job_id         TEXT,
    pushed_to_git  BOOLEAN NOT NULL DEFAULT false,
    pushed_at      TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_blocks_pushed   ON blocks (pushed_to_git, created_at);
"""


async def init_db() -> None:
    """Create the connection pool and run schema migrations."""
    global _pool
    url = os.environ.get("DATABASE_URL", "").strip()
    if not url:
        logger.warning("db_disabled_no_database_url")
        return
    _pool = await asyncpg.create_pool(url, min_size=1, max_size=4)
    async with _pool.acquire() as conn:
        await conn.execute(SCHEMA_SQL)
    logger.info("db_initialized")


async def close_db() -> None:
    global _pool
    if _pool:
        await _pool.close()
        _pool = None


def _pool_or_raise() -> asyncpg.Pool:
    if not _pool:
        raise RuntimeError("Database not initialized")
    return _pool


# ── Jobs ──────────────────────────────────────────────────────

async def create_job(*, research_goal: str, config: dict[str, Any], idempotency_key: str = "") -> dict:
    pool = _pool_or_raise()
    now = datetime.now(timezone.utc)
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """INSERT INTO jobs (job_id, created_at, updated_at, research_goal, status, config, idempotency_key)
               VALUES ($1, $2, $2, $3, 'pending', $4::jsonb, $5)
               RETURNING *""",
            _ulid(now),
            now,
            research_goal,
            json.dumps(config, default=str),
            idempotency_key or None,
        )
    return _row_to_dict(row)


async def find_by_idempotency_key(key: str) -> dict | None:
    pool = _pool_or_raise()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT * FROM jobs WHERE idempotency_key = $1", key
        )
    return _row_to_dict(row) if row else None


async def update_job(
    job_id: str, *, status: str | None = None, result: dict | None = None,
    error: str | None = None,
) -> dict | None:
    pool = _pool_or_raise()
    now = datetime.now(timezone.utc)
    sets: list[str] = ["updated_at = $1"]
    args: list[Any] = [now]
    idx = 2
    if status is not None:
        sets.append(f"status = ${idx}"); args.append(status); idx += 1
    if result is not None:
        sets.append(f"result = ${idx}::jsonb"); args.append(json.dumps(result, default=str)); idx += 1
    if error is not None:
        sets.append(f"error = ${idx}"); args.append(error); idx += 1
    args.append(job_id)
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            f"UPDATE jobs SET {', '.join(sets)} WHERE job_id = ${idx} RETURNING *",
            *args,
        )
    return _row_to_dict(row) if row else None


async def append_event(job_id: str, *, stage: str, message: str, data: dict | None = None) -> dict | None:
    pool = _pool_or_raise()
    now = datetime.now(timezone.utc)
    event = {"at": now.isoformat(), "stage": stage, "message": message, "data": data}
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """UPDATE jobs SET updated_at = $1,
               events = events || $2::jsonb
               WHERE job_id = $3
               RETURNING *""",
            now, json.dumps([event], default=str), job_id,
        )
    return _row_to_dict(row) if row else None


async def get_job(job_id: str) -> dict | None:
    pool = _pool_or_raise()
    async with pool.acquire() as conn:
        row = await conn.fetchrow("SELECT * FROM jobs WHERE job_id = $1", job_id)
    return _row_to_dict(row) if row else None


async def list_jobs(limit: int = 50, status: str | None = None) -> list[dict]:
    pool = _pool_or_raise()
    if status:
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM jobs WHERE status = $1 ORDER BY created_at DESC LIMIT $2",
                status, limit,
            )
    else:
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM jobs ORDER BY created_at DESC LIMIT $1", limit,
            )
    return [_row_to_dict(r) for r in rows]


# ── Blocks ────────────────────────────────────────────────────

async def save_block(
    *, block_number: int, research_goal: str, paper_title: str,
    paper_md: str, paper_html: str, analysis: dict, market_price: float,
    simulation_html: str = "", job_id: str = "",
) -> None:
    pool = _pool_or_raise()
    now = datetime.now(timezone.utc)
    async with pool.acquire() as conn:
        await conn.execute(
            """INSERT INTO blocks (block_number, created_at, research_goal, paper_title,
               paper_md, paper_html, analysis_json, market_price, simulation_html, job_id)
               VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb, $8, $9, $10)
               ON CONFLICT (block_number) DO UPDATE SET
               paper_title = EXCLUDED.paper_title,
               paper_md = EXCLUDED.paper_md,
               paper_html = EXCLUDED.paper_html,
               analysis_json = EXCLUDED.analysis_json,
               market_price = EXCLUDED.market_price,
               simulation_html = EXCLUDED.simulation_html,
               updated_at = $2""",
            block_number, now, research_goal, paper_title,
            paper_md, paper_html, json.dumps(analysis, default=str),
            market_price, simulation_html, job_id or None,
        )


async def get_block(block_number: int) -> dict | None:
    pool = _pool_or_raise()
    async with pool.acquire() as conn:
        row = await conn.fetchrow("SELECT * FROM blocks WHERE block_number = $1", block_number)
    return _row_to_dict(row) if row else None


async def get_latest_block() -> dict | None:
    pool = _pool_or_raise()
    async with pool.acquire() as conn:
        row = await conn.fetchrow("SELECT * FROM blocks ORDER BY block_number DESC LIMIT 1")
    return _row_to_dict(row) if row else None


async def list_blocks(limit: int = 30) -> list[dict]:
    pool = _pool_or_raise()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT * FROM blocks ORDER BY block_number DESC LIMIT $1", limit,
        )
    return [_row_to_dict(r) for r in rows]


async def get_unpushed_blocks() -> list[dict]:
    pool = _pool_or_raise()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT * FROM blocks WHERE pushed_to_git = false ORDER BY block_number ASC"
        )
    return [_row_to_dict(r) for r in rows]


async def mark_blocks_pushed(block_numbers: list[int]) -> None:
    pool = _pool_or_raise()
    now = datetime.now(timezone.utc)
    async with pool.acquire() as conn:
        await conn.execute(
            "UPDATE blocks SET pushed_to_git = true, pushed_at = $1 WHERE block_number = ANY($2::int[])",
            now, block_numbers,
        )


async def get_next_block_number() -> int:
    pool = _pool_or_raise()
    async with pool.acquire() as conn:
        row = await conn.fetchrow("SELECT COALESCE(MAX(block_number), 0) + 1 AS next FROM blocks")
    return row["next"] if row else 1


# ── Hydration ─────────────────────────────────────────────────

async def hydrate_blocks_from_db(results_dir: str) -> int:
    """Write all blocks from the database into the local results/ directory.
    Called at startup so the Vercel static site has block files.
    Returns count of blocks hydrated."""
    import shutil
    from pathlib import Path

    blocks = await list_blocks(100)
    if not blocks:
        return 0
    base = Path(results_dir)
    base.mkdir(parents=True, exist_ok=True)
    for b in blocks:
        bn = b["block_number"]
        d = base / f"block-{bn}"
        d.mkdir(parents=True, exist_ok=True)
        (d / "paper.md").write_text(b.get("paper_md", ""), encoding="utf-8")
        (d / "paper.html").write_text(b.get("paper_html", ""), encoding="utf-8")
        analysis = b.get("analysis_json") or {}
        if isinstance(analysis, str):
            import json as _json
            analysis = _json.loads(analysis)
        block_json = {
            "block_number": bn,
            "created_at": str(b.get("created_at", "")),
            "research_goal": b.get("research_goal", ""),
            "paper_title": b.get("paper_title", ""),
            "market_price": b.get("market_price", 0),
            "models": (analysis or {}).get("models", {}),
            "n_submitters": (analysis or {}).get("n_submitters", 3),
            "stats": (analysis or {}).get("stats", {}),
        }
        (d / "block.json").write_text(json.dumps(block_json, indent=2), encoding="utf-8")
        (d / "analysis.json").write_text(json.dumps(analysis, indent=2, default=str), encoding="utf-8")
    return len(blocks)


# ── Helpers ───────────────────────────────────────────────────

def _row_to_dict(row: Any) -> dict:
    if row is None:
        return {}
    d = dict(row)
    for col in ("config", "result", "events", "analysis_json"):
        if col in d and isinstance(d[col], str):
            try:
                d[col] = json.loads(d[col])
            except (json.JSONDecodeError, TypeError):
                pass
    if "created_at" in d and hasattr(d["created_at"], "isoformat"):
        d["created_at"] = d["created_at"].isoformat()
    if "updated_at" in d and hasattr(d["updated_at"], "isoformat"):
        d["updated_at"] = d["updated_at"].isoformat()
    if "pushed_at" in d and hasattr(d["pushed_at"], "isoformat"):
        d["pushed_at"] = d["pushed_at"].isoformat()
    return d


def _ulid(now: datetime) -> str:
    """Simple ULID-like id: timestamp + random hex."""
    ts = int(now.timestamp() * 1000)
    import random
    rand = "".join(random.choice("0123456789ABCDEFGHJKMNPQRSTVWXYZ") for _ in range(12))
    return f"{ts:013X}{rand}"
