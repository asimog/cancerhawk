"""CancerHawk fused engine — FastAPI app on localhost:8765.

Serves:
  GET  /              → web/index.html
  GET  /static/*      → web/ assets
  GET  /api/models    → curated OpenRouter model list for dropdowns
  WS   /ws/hermes/run → run a Hermes-supervised block

Run:
  python app/main.py
  open http://localhost:8765
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import BackgroundTasks, FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# Configure structured logging with millisecond timestamps (MOTO-style)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d | %(levelname)-5s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
GIVEWELL_SOLANA = "4Z2DBVoQCJZ42cCTDMNvYDUqRjA1C3vV7B155Mc6jGah"

logger = logging.getLogger("cancerhawk")


def _resolve_wallet(raw: Any) -> str:
    """Resolve wallet to Solana address. Defaults to GiveWell charity."""
    wallet = str(raw or "").strip()
    if wallet:
        if len(wallet) >= 32 and all(c in "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz" for c in wallet[:44]):
            return wallet
    return GIVEWELL_SOLANA

APP_DIR = Path(__file__).resolve().parent

from .token_tracker import APICall, APIFailureLimitExceeded, TokenTracker  # noqa: E402
from .hermes_supervisor import HermesRunConfig, HermesSupervisor  # noqa: E402
from .openrouter import close as close_openrouter  # noqa: E402
from .jobs import append_job_event, create_job, find_job_by_idempotency_key, get_job, hydrate_from_history, job_store_info, list_jobs, update_job_status  # noqa: E402
from .publisher import publish_from_staging, STAGING_DIR  # noqa: E402
from .agents import estimate_run_cost, get_leaderboard, get_prompts, parse_agent_run, submit_paper  # noqa: E402
from . import db  # noqa: E402

HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "8765"))
TERMINAL_JOB_STATUSES = {"completed", "published", "failed", "stopped"}


class JobStopped(RuntimeError):
    pass

app = FastAPI(title="CancerHawk")

# CORS allowlist — set CANCERHAWK_CORS_ORIGINS as a comma-separated list to
# add the Vercel and any custom origins on Railway. Default keeps GH Pages +
# localhost dev working.
_default_cors = (
    "https://asimog.github.io,"
    "https://asimog.github.io/cancerhawk,"
    "https://cancerhawk.vercel.app,"
    "http://localhost:8765,"
    "http://localhost:3000,"
    "http://127.0.0.1:8765"
)
_cors_origins = [
    o.strip() for o in os.environ.get("CANCERHAWK_CORS_ORIGINS", _default_cors).split(",")
    if o.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=str(APP_DIR / "web")), name="static")
RESULTS_DIR = APP_DIR.parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/results", StaticFiles(directory=str(RESULTS_DIR), html=True), name="results")


@app.get("/")
async def root() -> FileResponse:
    return FileResponse(str(APP_DIR / "web" / "index.html"))


@app.get("/api/models")
async def models() -> JSONResponse:
    return JSONResponse({"models": MODELS, "defaults": DEFAULT_MODELS})


@app.get("/api/healthcheck")
async def healthcheck() -> JSONResponse:
    return JSONResponse({"status": "ok", "service": "cancerhawk"})


@app.get("/api/health")
async def health() -> JSONResponse:
    return JSONResponse({"status": "ok", "service": "cancerhawk"})


@app.get("/api/hermes/status")
async def hermes_status() -> JSONResponse:
    """Expose whether Railway is configured for autonomous Hermes publishing."""
    return JSONResponse({
        "service": "cancerhawk-hermes",
        "runs_on_railway_process": True,
        "supervises": ["moto", "analysis", "miroshark_peer_review", "simulation_generation", "repo_publish"],
        "github_repo": os.environ.get("GITHUB_REPO", ""),
        "github_branch": os.environ.get("GITHUB_BRANCH", "master"),
        "commit_paths": [p.strip() for p in os.environ.get("HERMES_COMMIT_PATHS", "results").split(",") if p.strip()],
        "vercel_deploy_hook": bool(os.environ.get("VERCEL_DEPLOY_HOOK_URL", "").strip()),
        "job_store": job_store_info(),
    })


@app.get("/api/jobs")
async def get_jobs(limit: int = 50, status: str = None) -> JSONResponse:
    """List jobs, newest first."""
    jobs = list_jobs(limit=limit, status=status)
    return JSONResponse({"jobs": jobs})


@app.get("/api/jobs/{job_id}")
async def get_job_details(job_id: str) -> JSONResponse:
    """Return a single job by its ID."""
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    return JSONResponse(job)


@app.post("/api/jobs/{job_id}/stop")
async def stop_job(job_id: str) -> JSONResponse:
    """Request a running job to stop and mark its card immediately."""
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    if job.get("status") in TERMINAL_JOB_STATUSES:
        return JSONResponse({"job": job, "job_id": job_id, "stopped": False, "status": job.get("status")})

    message = "Stopped by user request."
    update_job_status(job_id, "stopped", error=message)
    append_job_event(job_id, stage="stopped", message=message, data={"job_id": job_id})
    stopped_job = get_job(job_id) or job
    return JSONResponse({"job": stopped_job, "job_id": job_id, "stopped": True, "status": "stopped"})


@app.post("/api/jobs/start")
async def start_job(payload: dict[str, Any], background_tasks: BackgroundTasks) -> JSONResponse:
    """Create a job card immediately, then run CancerHawk in the background."""
    api_key, research_goal, n_submitters, auto_publish, git_push, models_cfg, enable_paysh = _parse_run_payload_or_400(payload)

    idempotency_key = str(payload.get("idempotency_key") or "").strip()[:160]
    if idempotency_key:
        existing_job = find_job_by_idempotency_key(idempotency_key)
        if existing_job:
            return JSONResponse({"job": existing_job, "job_id": existing_job["job_id"], "deduped": True})

    job_config = {
        "models": models_cfg,
        "n_submitters": n_submitters,
        "auto_publish": auto_publish,
        "git_push": git_push,
        "idempotency_key": idempotency_key or None,
        "wallet_address": _resolve_wallet(payload.get("wallet_address")),
        "enable_paysh": enable_paysh,
    }
    job = create_job(research_goal=research_goal, config=job_config)
    job_id = job["job_id"]
    update_job_status(job_id, "running")
    append_job_event(
        job_id,
        stage="start",
        message=f"Starting block · goal: {research_goal[:120]}",
        data={"models": models_cfg, "job_id": job_id, "enable_paysh": enable_paysh},
    )
    background_tasks.add_task(
        _run_job_background,
        job_id,
        api_key,
        research_goal,
        n_submitters,
        auto_publish,
        git_push,
        models_cfg,
        enable_paysh,
    )
    job = get_job(job_id) or job
    logger.info("job_created", extra={"job_id": job_id, "goal": research_goal[:120]})
    return JSONResponse({"job": job, "job_id": job_id})


@app.get("/api/blocks/{block_number}")
async def block_bundle(block_number: int) -> JSONResponse:
    """Return the locally published paper, peer review, and simulations bundle."""
    if block_number < 1:
        raise HTTPException(status_code=404, detail="block not found")

    block_dir = RESULTS_DIR / f"block-{block_number}"
    meta_path = block_dir / "block.json"
    analysis_path = block_dir / "analysis.json"
    paper_path = block_dir / "paper.md"

    if not meta_path.exists() or not analysis_path.exists() or not paper_path.exists():
        raise HTTPException(status_code=404, detail="block not found")

    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
        paper_md = paper_path.read_text(encoding="utf-8")
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("block_bundle_read_failed", extra={"block": block_number, "error": str(exc)})
        raise HTTPException(status_code=500, detail="block bundle could not be loaded") from exc

    return JSONResponse({
        "block": block_number,
        "meta": meta,
        "paper_md": paper_md,
        "peer_reviews": analysis.get("peer_reviews", []),
        "simulations": analysis.get("simulations", []),
        "analysis": {
            "market_price": analysis.get("market_price"),
            "consensus_dim": analysis.get("consensus_dim"),
            "headline_catalysts": analysis.get("headline_catalysts", []),
            "topics": analysis.get("topics", []),
        },
    })


# ---------------------------------------------------------------------------
# Agent API — External AI agents can submit papers, run pipelines, check costs
# ---------------------------------------------------------------------------

@app.get("/api/agents/prompts")
async def agent_prompts() -> JSONResponse:
    """Expose prompt templates for agents running papers locally (BYOK)."""
    return JSONResponse(get_prompts())


@app.post("/api/agents/cost")
async def agent_cost(payload: dict[str, Any]) -> JSONResponse:
    """Estimate cost for a run. Supports mode: openrouter, paysh_cancerhawk, paysh_owned, local."""
    model = str(payload.get("model") or "openrouter/free")
    n_submitters = int(payload.get("n_submitters") or 3)
    mode = str(payload.get("mode") or "openrouter")
    return JSONResponse(estimate_run_cost(model=model, n_submitters=n_submitters, mode=mode))


@app.post("/api/agents/submit")
async def agent_submit(payload: dict[str, Any]) -> JSONResponse:
    """Accept a paper submission from an external agent."""
    try:
        agent_name = str(payload.get("agent_name") or "").strip()
        if not agent_name:
            raise ValueError("agent_name is required")
        paper_title = str(payload.get("paper_title") or "").strip()
        if not paper_title:
            raise ValueError("paper_title is required")
        paper_content = str(payload.get("paper_content") or "").strip()
        if not paper_content:
            raise ValueError("paper_content is required")
        research_goal = str(payload.get("research_goal") or "").strip()
        if not research_goal:
            raise ValueError("research_goal is required")

        result = submit_paper(
            agent_name=agent_name,
            paper_title=paper_title,
            paper_content=paper_content,
            research_goal=research_goal,
            agent_model=str(payload.get("agent_model") or "unknown"),
            peer_reviews=payload.get("peer_reviews"),
            simulations=payload.get("simulations"),
            wallet_address=str(payload.get("wallet_address") or "").strip() or GIVEWELL_SOLANA,
        )
        return JSONResponse(result)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/agents/run")
async def agent_run(payload: dict[str, Any], background_tasks: BackgroundTasks) -> JSONResponse:
    """Run the full CancerHawk pipeline.

    Modes:
      - openrouter       : Agent provides OpenRouter key (default)
      - paysh_cancerhawk : Agent pays CancerHawk via pay.sh; CancerHawk handles everything
      - paysh_owned      : Agent uses own pay.sh wallet; provides OpenRouter key
      - local            : Download prompts, run locally, submit result
    """
    try:
        api_key, research_goal, model, n_submitters, agent_name, mode = parse_agent_run(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if mode == "local":
        return JSONResponse({
            "mode": "local",
            "message": "Use GET /api/agents/prompts to download prompt templates, run locally, then POST /api/agents/submit with your paper.",
            "next_step": "GET /api/agents/prompts",
        })

    enable_paysh = mode == "paysh_cancerhawk"

    models_cfg = {
        "submitter": model,
        "validator": model,
        "compiler": model,
        "archetype": model,
        "topic_deriver": model,
    }

    job_config = {
        "models": models_cfg,
        "n_submitters": n_submitters,
        "auto_publish": True,
        "git_push": False,
        "agent_name": agent_name,
        "mode": mode,
        "enable_paysh": enable_paysh,
    }
    job = create_job(research_goal=research_goal, config=job_config)
    job_id = job["job_id"]
    update_job_status(job_id, "running")
    append_job_event(
        job_id, stage="start",
        message=f"Agent run [{mode}] · {agent_name or 'anonymous'} · goal: {research_goal[:120]}",
        data={"models": models_cfg, "job_id": job_id, "agent_name": agent_name, "mode": mode, "enable_paysh": enable_paysh},
    )

    background_tasks.add_task(
        _run_job_background, job_id, api_key, research_goal,
        n_submitters, True, False, models_cfg, enable_paysh,
    )

    job = get_job(job_id) or job
    logger.info("agent_run_started", extra={"job_id": job_id, "agent": agent_name, "goal": research_goal[:120], "mode": mode})
    return JSONResponse({
        "job": job,
        "job_id": job_id,
        "message": f"Pipeline started [{mode}]. Track progress at /api/jobs/{job_id}",
        "cost_estimate": estimate_run_cost(model=model, n_submitters=n_submitters, mode=mode),
    })


@app.get("/api/agents/leaderboard")
async def agent_leaderboard() -> JSONResponse:
    """Return the list of agent paper submissions."""
    return JSONResponse({
        "submissions": get_leaderboard(),
        "total": len(get_leaderboard()),
        "prize": "0.01 USDC",
        "note": "Highest market-price synthesis wins. Submissions are peer-reviewed by CancerHawk archetype engines.",
    })


@app.post("/api/agents/enrich")
async def agent_enrich(payload: dict[str, Any]) -> JSONResponse:
    """Research enrichment via pay.sh endpoints. Returns web search results for a goal."""
    from .paysh import research_enrichment, PAYSH_ENDPOINTS
    research_goal = str(payload.get("research_goal") or "").strip()
    if not research_goal:
        raise HTTPException(status_code=400, detail="research_goal is required")
    enrichment = research_enrichment(research_goal)
    return JSONResponse({
        "research_goal": research_goal[:200],
        "enrichment": enrichment,
        "available_endpoints": list(PAYSH_ENDPOINTS.keys()),
    })


@app.websocket("/ws/run")
async def ws_run(ws: WebSocket) -> None:
    await _ws_hermes_run(ws)


@app.websocket("/ws/hermes/run")
async def ws_hermes_run(ws: WebSocket) -> None:
    await _ws_hermes_run(ws)


def _parse_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    raise ValueError("expected boolean")


def _parse_run_payload(cfg: dict[str, Any]) -> tuple[str, str, int, bool, bool, dict[str, str], bool]:
    user_api_key = (cfg.get("api_key") or "").strip()
    server_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    user_provided_key = bool(user_api_key and user_api_key != server_key)
    api_key = user_api_key or server_key
    research_goal = (cfg.get("research_goal") or "").strip()
    n_submitters = int(cfg.get("n_submitters") or 3)
    if n_submitters < 1 or n_submitters > 8:
        raise ValueError("n_submitters must be between 1 and 8")
    auto_publish = _parse_bool(cfg.get("auto_publish"), True)
    git_push = _parse_bool(cfg.get("git_push"), True)
    enable_paysh = _parse_bool(cfg.get("enable_paysh"), False)
    models_cfg = {
        "submitter": _resolve_job_model(cfg.get("submitter"), "submitter", user_provided_key),
        "validator": _resolve_job_model(cfg.get("validator"), "validator", user_provided_key),
        "compiler": _resolve_job_model(cfg.get("compiler"), "compiler", user_provided_key),
        "archetype": _resolve_job_model(cfg.get("archetype"), "archetype", user_provided_key),
        "topic_deriver": _resolve_job_model(cfg.get("topic_deriver"), "topic_deriver", user_provided_key),
    }
    return api_key, research_goal, n_submitters, auto_publish, git_push, models_cfg, enable_paysh


def _resolve_job_model(value: Any, role: str, user_provided_key: bool = False) -> str:
    configured = str(value or "").strip()
    if user_provided_key and configured and configured in MODELS:
        return configured
    return FREE_ROUTER_MODEL


def _parse_run_payload_or_400(cfg: dict[str, Any]) -> tuple[str, str, int, bool, bool, dict[str, str], bool]:
    try:
        api_key, research_goal, n_submitters, auto_publish, git_push, models_cfg, enable_paysh = _parse_run_payload(cfg)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"Invalid run payload: {exc}") from exc
    if not api_key:
        raise HTTPException(status_code=400, detail="OpenRouter API key required")
    if not research_goal:
        raise HTTPException(status_code=400, detail="Research goal required")
    if len(research_goal) > 1000:
        raise HTTPException(status_code=400, detail="Research goal must be at most 1000 characters")
    research_goal = _sanitize_goal(research_goal)
    return api_key, research_goal, n_submitters, auto_publish, git_push, models_cfg, enable_paysh


def _raise_if_job_stopped(job_id: str) -> None:
    job = get_job(job_id)
    if job and job.get("status") == "stopped":
        raise JobStopped("Stopped by user request.")


async def _run_job_background(
    job_id: str,
    api_key: str,
    research_goal: str,
    n_submitters: int,
    auto_publish: bool,
    git_push: bool,
    models_cfg: dict[str, str],
    enable_paysh: bool = False,
) -> None:
    tracker = TokenTracker()
    run_start = time.time()

    async def emit(stage: str, message: str, data: dict | None = None) -> None:
        _raise_if_job_stopped(job_id)
        append_job_event(job_id, stage=stage, message=message, data=data)
        logger.info("pipeline_event", extra={"job_id": job_id, "stage": stage, "event_message": message[:200], "data": data})

    async def on_call(call: APICall) -> None:
        message = (
            f"#{call.seq} {call.role} · {call.model} · "
            f"in={call.prompt_tokens} out={call.completion_tokens} "
            f"({call.latency_ms}ms, ${call.cost_usd:.4f})"
            + ("" if call.ok else f" · ERR {call.error[:80] if call.error else ''}")
        )
        append_job_event(
            job_id,
            stage="api_call",
            message=message,
            data={"call": call.to_dict(), "totals": tracker.stats()},
        )
        _raise_if_job_stopped(job_id)
        logger.info(
            "api_call #%s role=%s model=%s in=%s out=%s latency=%sms cost=$%.4f",
            call.seq,
            call.role,
            call.model,
            call.prompt_tokens,
            call.completion_tokens,
            call.latency_ms,
            call.cost_usd,
        )

    try:
        await emit(
            "hermes",
            "Hermes supervisor started: job card is now live",
            {"models": models_cfg, "job_id": job_id},
        )
        supervisor = HermesSupervisor(emit=emit, on_call=on_call, tracker=tracker)
        result = await supervisor.run(
            HermesRunConfig(
                api_key=api_key,
                research_goal=research_goal,
                models=models_cfg,
                n_submitters=n_submitters,
                auto_publish=False,  # stage for later publication
                git_push=git_push,
                job_id=job_id,
                stage=True,
                enable_paysh=enable_paysh,
            )
        )
        _raise_if_job_stopped(job_id)
        update_job_status(job_id, "completed", result={
            "title": result.title,
            "market_price": result.market_price,
            "block": result.block,
            "result_url": result.result_url,
            "stats": result.stats,
            "calls": [c for c in result.calls],
            "git_status": result.git_status,
        })
        stats = result.stats
        await emit(
            "done",
            (
                f"✓ Hermes complete · market price = {result.market_price:.2f} · "
                f"{stats['total_calls']} API calls · "
                f"{stats['total_tokens']:,} tokens · "
                f"${stats['total_cost_usd']:.4f} · "
                f"{stats['elapsed_seconds']:.0f}s"
            ),
            {
                "title": result.title,
                "market_price": result.market_price,
                "block": result.block,
                "result_url": result.result_url,
                "stats": stats,
                "calls": result.calls,
                "git_status": result.git_status,
            },
        )
    except JobStopped as exc:
        append_job_event(job_id, stage="stopped", message=str(exc), data={"failed_calls": tracker.failed_calls})
        update_job_status(job_id, "stopped", error=str(exc))
    except APIFailureLimitExceeded as exc:
        message = str(exc)
        append_job_event(job_id, stage="stopped", message=message, data={"failed_calls": exc.failed_calls, "limit": exc.limit})
        update_job_status(job_id, "failed", error=message)
    except Exception as exc:
        tb = traceback.format_exc()
        logger.error("job_run_failed", extra={"job_id": job_id, "error_type": type(exc).__name__, "error": str(exc)})
        append_job_event(job_id, stage="error", message=f"{type(exc).__name__}: {exc}", data={"traceback": tb})
        update_job_status(job_id, "failed", error=f"{type(exc).__name__}: {str(exc)[:500]}")
    finally:
        logger.info("job_run_ended", extra={"job_id": job_id, "run_elapsed_seconds": round(time.time() - run_start, 2)})


def _sanitize_goal(goal: str) -> str:
    """Strip HTML tags, null bytes, and control characters from research goals."""
    import re
    goal = goal.replace("\x00", "")
    goal = re.sub(r"<[^>]*>", "", goal)
    goal = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", goal)
    return goal.strip()


async def _ws_hermes_run(ws: WebSocket) -> None:
    await ws.accept()
    run_start = time.time()

    try:
        cfg_text = await ws.receive_text()
        cfg = json.loads(cfg_text)
    except Exception as exc:
        logger.error("bad_config", extra={"error": str(exc)})
        await ws.send_text(json.dumps({"stage": "error", "message": f"bad config: {exc}"}))
        await ws.close()
        return

    try:
        api_key, research_goal, n_submitters, auto_publish, git_push, models_cfg, _ = _parse_run_payload(cfg)
    except Exception as exc:
        await ws.send_text(json.dumps({"stage": "error", "message": f"bad config: {exc}"}))
        await ws.close()
        return

    if not api_key:
        logger.warning("missing_api_key")
        await ws.send_text(json.dumps({"stage": "error", "message": "OpenRouter API key required"}))
        await ws.close()
        return
    if not research_goal:
        logger.warning("missing_research_goal")
        await ws.send_text(json.dumps({"stage": "error", "message": "Research goal required"}))
        await ws.close()
        return
    if len(research_goal) > 1000:
        await ws.send_text(json.dumps({"stage": "error", "message": "Research goal must be at most 1000 characters"}))
        await ws.close()
        return

    research_goal = _sanitize_goal(research_goal)

    # Create a job record for this run
    job_config = {"models": models_cfg, "n_submitters": n_submitters, "auto_publish": auto_publish, "git_push": git_push}
    job = create_job(research_goal=research_goal, config=job_config)
    job_id = job["job_id"]
    update_job_status(job_id, "running")
    logger.info("job_created", extra={"job_id": job_id, "goal": research_goal[:120]})

    logger.info(
        "run_start",
        extra={
            "goal": research_goal[:120],
            "models": models_cfg,
            "n_submitters": n_submitters,
            "auto_publish": auto_publish,
            "git_push": git_push,
        },
    )
    append_job_event(
        job_id,
        stage="start",
        message=f"Starting block · goal: {research_goal[:120]}",
        data={"models": models_cfg, "job_id": job_id},
    )
    await ws.send_text(json.dumps({"stage": "start", "message": f"Starting block · goal: {research_goal[:120]}", "data": {"models": models_cfg, "job_id": job_id}}))

    tracker = TokenTracker()

    async def emit(stage: str, message: str, data: dict | None = None) -> None:
        try:
            payload = {"stage": stage, "message": message, "data": data}
            append_job_event(job_id, stage=stage, message=message, data=data)
            await ws.send_text(json.dumps(payload))
            logger.info("pipeline_event", extra={"stage": stage, "event_message": message[:200], "data": data})
        except Exception as exc:
            logger.warning("emit_failed", extra={"error": str(exc)})

    async def on_call(call: APICall) -> None:
        try:
            await ws.send_text(json.dumps({
                "stage": "api_call",
                "message": (
                    f"#{call.seq} {call.role} · {call.model} · "
                    f"in={call.prompt_tokens} out={call.completion_tokens} "
                    f"({call.latency_ms}ms, ${call.cost_usd:.4f})"
                    + ("" if call.ok else f" · ERR {call.error[:80] if call.error else ''}")
                ),
                "data": {"call": call.to_dict(), "totals": tracker.stats()},
            }))
            append_job_event(
                job_id,
                stage="api_call",
                message=(
                    f"#{call.seq} {call.role} · {call.model} · "
                    f"in={call.prompt_tokens} out={call.completion_tokens} "
                    f"({call.latency_ms}ms, ${call.cost_usd:.4f})"
                    + ("" if call.ok else f" · ERR {call.error[:80] if call.error else ''}")
                ),
                data={"call": call.to_dict(), "totals": tracker.stats()},
            )
            # Also log to server console with an informative single-line message
            log_msg = (
                f"api_call #{call.seq} role={call.role} model={call.model} "
                f"in={call.prompt_tokens} out={call.completion_tokens} "
                f"latency={call.latency_ms}ms cost=${call.cost_usd:.4f}"
            )
            if not call.ok:
                log_msg += f" ERROR={call.error[:80] if call.error else 'unknown'}"
            logger.info(log_msg)
        except Exception as exc:
            logger.warning("on_call_emit_failed", extra={"error": str(exc)})

    # Main pipeline wrapped in try/except/finally
    try:
        supervisor = HermesSupervisor(emit=emit, on_call=on_call, tracker=tracker)
        result = await supervisor.run(
            HermesRunConfig(
                api_key=api_key,
                research_goal=research_goal,
                models=models_cfg,
                n_submitters=n_submitters,
                auto_publish=False,  # stage for later publication
                git_push=git_push,
                job_id=job_id,
                stage=True,
                enable_paysh=False,
            )
        )
        # Update job with result
        update_job_status(job_id, "completed", result={
            "title": result.title,
            "market_price": result.market_price,
            "block": result.block,
            "result_url": result.result_url,
            "stats": result.stats,
            "calls": result.calls,
            "git_status": result.git_status,
        })
        stats = result.stats
        logger.info(
            "run_complete",
            extra={
                "title": result.title,
                "market_price": result.market_price,
                "total_calls": stats["total_calls"],
                "total_tokens": stats["total_tokens"],
                "total_cost_usd": stats["total_cost_usd"],
                "elapsed_seconds": stats["elapsed_seconds"],
                "block": result.block,
                "git_status": result.git_status,
            },
        )
        await ws.send_text(json.dumps({
            "stage": "done",
            "message": (
                f"✓ Hermes complete · market price = {result.market_price:.2f} · "
                f"{stats['total_calls']} API calls · "
                f"{stats['total_tokens']:,} tokens · "
                f"${stats['total_cost_usd']:.4f} · "
                f"{stats['elapsed_seconds']:.0f}s"
            ),
            "data": {
                "title": result.title,
                "market_price": result.market_price,
                "block": result.block,
                "result_url": result.result_url,
                "stats": stats,
                "calls": result.calls,
                "git_status": result.git_status,
            },
        }))
        append_job_event(
            job_id,
            stage="done",
            message=f"✓ Hermes complete · market price = {result.market_price:.2f}",
            data={"block": result.block, "result_url": result.result_url, "stats": stats},
        )
    except WebSocketDisconnect:
        logger.warning("client_disconnected")
        update_job_status(job_id, "failed", error="client disconnected")
        return
    except APIFailureLimitExceeded as exc:
        message = str(exc)
        logger.error("run_failed_api_failure_limit", extra={"failed_calls": exc.failed_calls, "limit": exc.limit})
        await emit("stopped", message, {"failed_calls": exc.failed_calls, "limit": exc.limit})
        update_job_status(job_id, "failed", error=message)
    except Exception as exc:
        tb = traceback.format_exc()
        logger.error("run_failed", extra={"error_type": type(exc).__name__, "error": str(exc)})
        await emit("error", f"{type(exc).__name__}: {exc}", {"traceback": tb})
        update_job_status(job_id, "failed", error=f"{type(exc).__name__}: {str(exc)[:500]}")
    finally:
        run_elapsed = time.time() - run_start
        logger.info("run_ended", extra={"run_elapsed_seconds": round(run_elapsed, 2)})
        try:
            await ws.close()
        except Exception:
            pass


# Background worker for periodic block publishing
async def publish_cycle_worker() -> None:
    while True:
        try:
            if STAGING_DIR.exists():
                candidates = []
                for job_dir in STAGING_DIR.iterdir():
                    if not job_dir.is_dir():
                        continue
                    job_id_cand = job_dir.name
                    meta_path = job_dir / "meta.json"
                    if not meta_path.is_file():
                        continue
                    try:
                        meta = json.loads(meta_path.read_text(encoding="utf-8"))
                        market_price = meta.get("market_price", 0.0)
                        timestamp = meta.get("timestamp", "")
                        candidates.append({
                            "job_id": job_id_cand,
                            "market_price": market_price,
                            "timestamp": timestamp,
                        })
                    except Exception as e:
                        logger.warning("failed_to_read_staging_meta", extra={"job_id": job_id_cand, "error": str(e)})
                        continue
                if candidates:
                    candidates.sort(key=lambda x: (-x["market_price"], x["timestamp"]))
                    for candidate in candidates:
                        try:
                            block_n = await asyncio.to_thread(publish_from_staging, candidate["job_id"])
                            logger.info("published_block_from_staging", extra={"job_id": candidate["job_id"], "block": block_n, "market_price": candidate["market_price"]})
                        except Exception as e:
                            logger.error("failed_to_publish_staged", extra={"job_id": candidate["job_id"], "error": str(e)})
            await asyncio.sleep(10)
            await maybe_auto_generate()
        except Exception as e:
            logger.error("publish_cycle_error", exc_info=e)
        await asyncio.sleep(600)


def _publish_cycle_enabled() -> bool:
    configured = os.environ.get("CANCERHAWK_PUBLISH_CYCLE_ENABLED", "").strip().lower()
    if configured:
        return configured in {"1", "true", "yes", "on"}

    return bool(os.environ.get("RAILWAY_ENVIRONMENT") or os.environ.get("RAILWAY_PROJECT_ID"))


def _git_sync_enabled() -> bool:
    return bool(os.environ.get("GITHUB_TOKEN") and os.environ.get("GITHUB_REPO"))


async def daily_git_sync_worker() -> None:
    """Push all unpushed blocks from DB to GitHub once per day."""
    while True:
        await asyncio.sleep(86400)  # 24 hours
        try:
            from .publisher import push_unpushed_to_git
            await asyncio.to_thread(push_unpushed_to_git)
        except Exception as e:
            logger.error("daily_git_sync_failed", exc_info=e)


async def _save_auto_block_to_db(result, goal: str, models: dict) -> None:
    """Save an auto-generated block to Postgres so it survives redeploys."""
    try:
        block_n = result.block
        block_dir = Path("results") / f"block-{block_n}"
        paper_md = (block_dir / "paper.md").read_text(encoding="utf-8") if (block_dir / "paper.md").exists() else ""
        paper_html = (block_dir / "paper.html").read_text(encoding="utf-8") if (block_dir / "paper.html").exists() else ""
        analysis = {}
        if (block_dir / "analysis.json").exists():
            analysis = json.loads((block_dir / "analysis.json").read_text(encoding="utf-8"))
        await db.save_block(
            block_number=block_n,
            research_goal=goal,
            paper_title=result.title or "",
            paper_md=paper_md,
            paper_html=paper_html,
            analysis=analysis,
            market_price=result.market_price or 0,
        )
    except Exception:
        pass

async def maybe_auto_generate() -> None:
    """Generate blocks automatically if no user submissions. Produces up to 3 blocks per cycle."""
    if os.environ.get("HERMES_AUTO_GENERATE_ENABLED", "").strip().lower() not in {"1", "true", "yes", "on"}:
        return

    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        return

    goals = [
        os.environ.get("HERMES_AUTO_GOAL", "Autonomous research: identify a promising oncology research direction"),
        os.environ.get("HERMES_AUTO_GOAL_2", "Autonomous research: mechanism for reversing T-cell exhaustion in solid tumors"),
        os.environ.get("HERMES_AUTO_GOAL_3", "Autonomous research: metabolic reprogramming of the tumor microenvironment"),
    ]
    n_to_generate = min(3, int(os.environ.get("HERMES_AUTO_BLOCKS_PER_CYCLE", "3")))

    for i in range(n_to_generate):
        models = {
            "submitter": os.environ.get("HERMES_MODEL_SUBMITTER", "openrouter/free"),
            "validator": os.environ.get("HERMES_MODEL_VALIDATOR", "openrouter/free"),
            "compiler": os.environ.get("HERMES_MODEL_COMPILER", "openrouter/free"),
            "archetype": os.environ.get("HERMES_MODEL_ARCHETYPE", "openrouter/free"),
            "topic_deriver": os.environ.get("HERMES_MODEL_TOPIC_DERIVER", "openrouter/free"),
        }
        n_submitters = int(os.environ.get("HERMES_N_SUBMITTERS", "3"))
        goal = goals[i % len(goals)]

        async def silent_emit(stage: str, message: str, data=None):
            logger.info(f"auto_gen [{stage}]: {message}")

        def silent_on_call(call):
            logger.debug(f"auto_gen API call: {call.role} {call.model}")

        supervisor = HermesSupervisor(emit=silent_emit, on_call=silent_on_call, tracker=TokenTracker())
        try:
            result = await supervisor.run(HermesRunConfig(
                api_key=api_key,
                research_goal=goal,
                models=models,
                n_submitters=n_submitters,
                auto_publish=True,
                git_push=False,
                stage=False,
                job_id=None,
                enable_paysh=False,
            ))
            logger.info("auto_generation_complete", extra={"block": result.block, "market_price": result.market_price, "goal": goal[:80]})
            await _save_auto_block_to_db(result, goal, models)
        except Exception as e:
            logger.error("auto_generation_failed", exc_info=e)

@app.on_event("startup")
async def startup_event() -> None:
    await db.init_db()
    hydrated = hydrate_from_history()
    if hydrated:
        logger.info("jobs_hydrated_from_history", extra={"count": hydrated})
    try:
        n = await db.hydrate_blocks_from_db(str(APP_DIR.parent / "results"))
        if n:
            logger.info("blocks_hydrated_from_db", extra={"count": n})
    except Exception:
        pass
    if _publish_cycle_enabled():
        asyncio.create_task(publish_cycle_worker())
        logger.info("publish_cycle_worker_started")
    else:
        logger.info("publish_cycle_worker_disabled")
    if _git_sync_enabled():
        asyncio.create_task(daily_git_sync_worker())


@app.on_event("shutdown")
async def shutdown() -> None:
    await close_openrouter()
    await db.close_db()


# Public job starts use OpenRouter's free auto-router. This avoids pinning a
# specific provider model at the beginning of a job, so a rate-limited free
# backend such as qwen/qwen3-coder:free does not become the job contract.
FREE_ROUTER_MODEL = "openrouter/free"
PAID_MODELS = [
    "deepseek/deepseek-v4-pro",
    "anthropic/claude-sonnet-4-20250514",
    "google/gemini-2.5-pro",
    "openai/gpt-4.1",
    "openai/o3",
    "anthropic/claude-3.5-haiku",
    "deepseek/deepseek-chat",
    "google/gemini-2.0-flash",
    "meta-llama/llama-4-maverick",
]
MODELS = PAID_MODELS + [FREE_ROUTER_MODEL]

DEFAULT_MODELS = {
    "submitter": FREE_ROUTER_MODEL,
    "validator": FREE_ROUTER_MODEL,
    "compiler": FREE_ROUTER_MODEL,
    "archetype": FREE_ROUTER_MODEL,
    "topic_deriver": FREE_ROUTER_MODEL,
}


if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")
