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
logger = logging.getLogger("cancerhawk")

APP_DIR = Path(__file__).resolve().parent

from .token_tracker import APICall, TokenTracker  # noqa: E402
from .hermes_supervisor import HermesRunConfig, HermesSupervisor  # noqa: E402
from .openrouter import close as close_openrouter  # noqa: E402
from .jobs import append_job_event, create_job, get_job, list_jobs, update_job_status  # noqa: E402

HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "8765"))

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
        "has_github_token": bool(os.environ.get("GITHUB_TOKEN", "").strip()),
        "commit_paths": [p.strip() for p in os.environ.get("HERMES_COMMIT_PATHS", "results").split(",") if p.strip()],
        "vercel_deploy_hook": bool(os.environ.get("VERCEL_DEPLOY_HOOK_URL", "").strip()),
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


@app.post("/api/jobs/start")
async def start_job(payload: dict[str, Any], background_tasks: BackgroundTasks) -> JSONResponse:
    """Create a job card immediately, then run CancerHawk in the background."""
    api_key, research_goal, n_submitters, auto_publish, git_push, models_cfg = _parse_run_payload(payload)
    if not api_key:
        raise HTTPException(status_code=400, detail="OpenRouter API key required")
    if not research_goal:
        raise HTTPException(status_code=400, detail="Research goal required")

    job_config = {
        "models": models_cfg,
        "n_submitters": n_submitters,
        "auto_publish": auto_publish,
        "git_push": git_push,
    }
    job = create_job(research_goal=research_goal, config=job_config)
    job_id = job["job_id"]
    update_job_status(job_id, "running")
    append_job_event(
        job_id,
        stage="start",
        message=f"Starting block · goal: {research_goal[:120]}",
        data={"models": models_cfg, "job_id": job_id},
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


@app.websocket("/ws/run")
async def ws_run(ws: WebSocket) -> None:
    await _ws_hermes_run(ws)


@app.websocket("/ws/hermes/run")
async def ws_hermes_run(ws: WebSocket) -> None:
    await _ws_hermes_run(ws)


def _parse_run_payload(cfg: dict[str, Any]) -> tuple[str, str, int, bool, bool, dict[str, str]]:
    api_key = (cfg.get("api_key") or "").strip()
    if not api_key:
        api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    research_goal = (cfg.get("research_goal") or "").strip()
    n_submitters = max(1, min(8, int(cfg.get("n_submitters") or 3)))
    auto_publish = bool(cfg.get("auto_publish", True))
    git_push = bool(cfg.get("git_push", True))
    models_cfg = {
        "submitter": cfg.get("submitter") or DEFAULT_MODELS["submitter"],
        "validator": cfg.get("validator") or DEFAULT_MODELS["validator"],
        "compiler": cfg.get("compiler") or DEFAULT_MODELS["compiler"],
        "archetype": cfg.get("archetype") or DEFAULT_MODELS["archetype"],
        "topic_deriver": cfg.get("topic_deriver") or DEFAULT_MODELS["topic_deriver"],
    }
    return api_key, research_goal, n_submitters, auto_publish, git_push, models_cfg


async def _run_job_background(
    job_id: str,
    api_key: str,
    research_goal: str,
    n_submitters: int,
    auto_publish: bool,
    git_push: bool,
    models_cfg: dict[str, str],
) -> None:
    tracker = TokenTracker()
    run_start = time.time()

    async def emit(stage: str, message: str, data: dict | None = None) -> None:
        append_job_event(job_id, stage=stage, message=message, data=data)
        logger.info("pipeline_event", extra={"job_id": job_id, "stage": stage, "message": message[:200], "data": data})

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
                auto_publish=auto_publish,
                git_push=git_push,
            )
        )
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
    except Exception as exc:
        tb = traceback.format_exc()
        logger.error("job_run_failed", extra={"job_id": job_id, "error_type": type(exc).__name__, "error": str(exc)})
        append_job_event(job_id, stage="error", message=f"{type(exc).__name__}: {exc}", data={"traceback": tb})
        update_job_status(job_id, "failed", error=f"{type(exc).__name__}: {str(exc)[:500]}")
    finally:
        logger.info("job_run_ended", extra={"job_id": job_id, "run_elapsed_seconds": round(time.time() - run_start, 2)})


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
        api_key, research_goal, n_submitters, auto_publish, git_push, models_cfg = _parse_run_payload(cfg)
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
            logger.info("pipeline_event", extra={"stage": stage, "message": message[:200], "data": data})
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
                auto_publish=auto_publish,
                git_push=git_push,
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


@app.on_event("shutdown")
async def shutdown() -> None:
    await close_openrouter()


# Free OpenRouter model list. The run UI reads this list from the Railway
# worker and offers every role the same selectable free options.
MODELS = [
    "openrouter/free",
    "openrouter/owl-alpha",
    "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
    "poolside/laguna-xs.2:free",
    "poolside/laguna-m.1:free",
    "inclusionai/ling-2.6-1t:free",
    "tencent/hy3-preview:free",
    "baidu/qianfan-ocr-fast:free",
    "google/gemma-4-26b-a4b-it:free",
    "google/gemma-4-31b-it:free",
    "google/lyria-3-pro-preview",
    "google/lyria-3-clip-preview",
    "nvidia/nemotron-3-super-120b-a12b:free",
    "minimax/minimax-m2.5:free",
    "liquid/lfm-2.5-1.2b-thinking:free",
    "liquid/lfm-2.5-1.2b-instruct:free",
    "nvidia/nemotron-3-nano-30b-a3b:free",
    "nvidia/nemotron-nano-12b-v2-vl:free",
    "qwen/qwen3-next-80b-a3b-instruct:free",
    "nvidia/nemotron-nano-9b-v2:free",
    "openai/gpt-oss-120b:free",
    "openai/gpt-oss-20b:free",
    "z-ai/glm-4.5-air:free",
    "qwen/qwen3-coder:free",
    "cognitivecomputations/dolphin-mistral-24b-venice-edition:free",
    "google/gemma-3n-e2b-it:free",
    "google/gemma-3n-e4b-it:free",
    "google/gemma-3-4b-it:free",
    "google/gemma-3-12b-it:free",
    "google/gemma-3-27b-it:free",
    "meta-llama/llama-3.3-70b-instruct:free",
    "meta-llama/llama-3.2-3b-instruct:free",
    "nousresearch/hermes-3-llama-3.1-405b:free",
]

DEFAULT_MODELS = {
    "submitter": "openrouter/free",
    "validator": "openrouter/free",
    "compiler": "openrouter/free",
    "archetype": "openrouter/free",
    "topic_deriver": "openrouter/free",
}


if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")
