"""Tests for PostgreSQL database module (app.db).

Requires DATABASE_URL or skips if Postgres is not available.
"""

import os
import json
import pytest

pytestmark = pytest.mark.skipif(
    not os.environ.get("DATABASE_URL"),
    reason="DATABASE_URL not set — Postgres tests skipped",
)


@pytest.fixture
async def db_conn():
    """Initialize the database pool and clean up after each test."""
    from app.db import init_db, close_db, _pool

    await init_db()
    assert _pool is not None

    # Clean tables
    async with _pool.acquire() as conn:
        await conn.execute("DELETE FROM blocks")
        await conn.execute("DELETE FROM jobs")

    yield

    async with _pool.acquire() as conn:
        await conn.execute("DELETE FROM blocks")
        await conn.execute("DELETE FROM jobs")
    await close_db()


@pytest.mark.asyncio
async def test_init_db_creates_pool(db_conn):
    from app.db import _pool
    assert _pool is not None


@pytest.mark.asyncio
async def test_create_and_get_job(db_conn):
    from app.db import create_job, get_job, update_job, append_event

    job = await create_job(
        research_goal="Test job",
        config={"n_submitters": 3},
    )
    assert "job_id" in job
    assert job["status"] == "pending"

    fetched = await get_job(job["job_id"])
    assert fetched is not None
    assert fetched["research_goal"] == "Test job"

    updated = await update_job(job["job_id"], status="running")
    assert updated["status"] == "running"

    with_event = await append_event(
        job["job_id"], stage="test", message="Test event", data={"key": "value"},
    )
    events = with_event.get("events") or []
    assert len(events) == 1
    assert events[0]["stage"] == "test"
    assert events[0]["message"] == "Test event"


@pytest.mark.asyncio
async def test_create_job_idempotency(db_conn):
    from app.db import create_job, find_by_idempotency_key

    await create_job(
        research_goal="Idempotent job",
        config={},
        idempotency_key="unique-key-123",
    )
    found = await find_by_idempotency_key("unique-key-123")
    assert found is not None
    assert found["research_goal"] == "Idempotent job"


@pytest.mark.asyncio
async def test_list_jobs(db_conn):
    from app.db import create_job, list_jobs

    await create_job(research_goal="Job 1", config={})
    await create_job(research_goal="Job 2", config={})

    jobs = await list_jobs(limit=10)
    assert len(jobs) >= 2
    assert jobs[0]["created_at"] >= jobs[1]["created_at"]  # descending


@pytest.mark.asyncio
async def test_update_job_result(db_conn):
    from app.db import create_job, update_job

    job = await create_job(research_goal="Result job", config={})
    updated = await update_job(
        job["job_id"],
        status="completed",
        result={"market_price": 0.75, "block": 42},
    )
    assert updated["status"] == "completed"
    assert updated["result"]["market_price"] == 0.75
    assert updated["result"]["block"] == 42


@pytest.mark.asyncio
async def test_update_job_error(db_conn):
    from app.db import create_job, update_job

    job = await create_job(research_goal="Error job", config={})
    updated = await update_job(
        job["job_id"], status="failed", error="something broke",
    )
    assert updated["status"] == "failed"
    assert updated["error"] == "something broke"


@pytest.mark.asyncio
async def test_save_and_get_block(db_conn):
    from app.db import save_block, get_block, get_next_block_number

    next_num = await get_next_block_number()
    assert next_num >= 1

    await save_block(
        block_number=99,
        research_goal="Block test",
        paper_title="Test Block 99",
        paper_md="# Test",
        paper_html="<h1>Test</h1>",
        analysis={"market_price": 0.5},
        market_price=0.5,
        job_id="test-job-123",
    )

    block = await get_block(99)
    assert block is not None
    assert block["paper_title"] == "Test Block 99"
    assert block["market_price"] == 0.5
    assert not block["pushed_to_git"]


@pytest.mark.asyncio
async def test_unpushed_blocks_and_mark_pushed(db_conn):
    from app.db import save_block, get_unpushed_blocks, mark_blocks_pushed

    for n in (50, 51, 52):
        await save_block(
            block_number=n, research_goal="Batch test",
            paper_title=f"Block {n}", paper_md="# T", paper_html="<p>T</p>",
            analysis={}, market_price=0.0,
        )

    unpushed = await get_unpushed_blocks()
    assert len(unpushed) >= 3

    await mark_blocks_pushed([50, 51, 52])
    still_unpushed = await get_unpushed_blocks()
    assert not any(b["block_number"] in (50, 51, 52) for b in still_unpushed)


@pytest.mark.asyncio
async def test_list_blocks(db_conn):
    from app.db import save_block, list_blocks

    for n in (200, 201):
        await save_block(
            block_number=n, research_goal="List test",
            paper_title=f"Block {n}", paper_md="# T", paper_html="<p>T</p>",
            analysis={}, market_price=0.0,
        )

    blocks = await list_blocks(limit=5)
    assert any(b["block_number"] in (200, 201) for b in blocks)
