"""CancerHawk fused engine — FastAPI app on localhost:8765.

Serves:
  GET  /              → web/index.html
  GET  /static/*      → web/ assets
  GET  /api/models    → curated OpenRouter model list for dropdowns
  WS   /ws/run        → run a full block (paper + analysis + publish)

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

import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
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

from .analysis_engine import run_analysis_engine  # noqa: E402
from .openrouter import chat_json, close as close_openrouter  # noqa: E402
from .paper_engine import run_paper_engine  # noqa: E402
from .prompts import topic_deriver_prompt  # noqa: E402
from .publisher import load_previous_block_context, publish_block, try_git_publish  # noqa: E402
from .simulation_engine import generate_html5_simulations  # noqa: E402
from .token_tracker import APICall, TokenTracker  # noqa: E402
from .peer_review_engine import (  # noqa: E402
    run_peer_review_engine,
    reviews_to_dict,
    consolidated_to_dict,
)

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

    api_key = (cfg.get("api_key") or "").strip()
    research_goal = (cfg.get("research_goal") or "").strip()
    n_submitters = max(1, min(8, int(cfg.get("n_submitters") or 3)))
    auto_publish = bool(cfg.get("auto_publish", True))
    git_push = bool(cfg.get("git_push", False))
    models_cfg = {
        "submitter": cfg.get("submitter") or DEFAULT_MODELS["submitter"],
        "validator": cfg.get("validator") or DEFAULT_MODELS["validator"],
        "compiler": cfg.get("compiler") or DEFAULT_MODELS["compiler"],
        "archetype": cfg.get("archetype") or DEFAULT_MODELS["archetype"],
        "topic_deriver": cfg.get("topic_deriver") or DEFAULT_MODELS["topic_deriver"],
    }

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
    await ws.send_text(json.dumps({"stage": "start", "message": f"Starting block · goal: {research_goal[:120]}", "data": {"models": models_cfg}}))

    tracker = TokenTracker()

    async def emit(stage: str, message: str, data: dict | None = None) -> None:
        try:
            payload = {"stage": stage, "message": message, "data": data}
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
        previous_block_context = load_previous_block_context()
        if previous_block_context:
            await emit(
                "prior_blocks",
                "Loaded prior CancerHawk blocks for optional citation/extension",
                {"chars": len(previous_block_context)},
            )

        # 1. Paper engine
        logger.info("stage_start", extra={"stage": "paper_engine"})
        paper = await run_paper_engine(
            api_key=api_key,
            research_goal=research_goal,
            models=models_cfg,
            n_submitters=n_submitters,
            emit=emit,
            tracker=tracker,
            on_call=on_call,
            previous_block_context=previous_block_context,
        )
        logger.info("stage_end", extra={"stage": "paper_engine"})
        paper_text = paper.full_text()
        logger.info(
            "paper_done",
            extra={
                "title": paper.title,
                "section_count": len(paper.sections),
                "word_count": len(paper_text.split()),
            },
        )
        await emit(
            "paper_done",
            f"Paper compiled: '{paper.title}' · {len(paper.sections)} sections · "
            f"{len(paper_text.split())} words · "
            f"aggregated {len(paper.accepted_submissions)} directions over "
            f"{getattr(paper, 'rounds_run', 0)} rounds "
            f"({getattr(paper, 'convergence_reason', '') or 'n/a'})",
            {
                "title": paper.title,
                "section_count": len(paper.sections),
                "accepted_count": len(paper.accepted_submissions),
                "rounds_run": getattr(paper, "rounds_run", 0),
                "convergence_reason": getattr(paper, "convergence_reason", ""),
            },
        )

        # 2. Analysis engine
        logger.info("stage_start", extra={"stage": "analysis_engine"})
        analysis = await run_analysis_engine(
            api_key=api_key,
            paper_text=paper_text,
            archetype_model=models_cfg["archetype"],
            emit=emit,
            tracker=tracker,
            on_call=on_call,
        )
        logger.info("stage_end", extra={"stage": "analysis_engine"})

        # 3. Peer review — MiroShark agents evaluate the paper
        logger.info("stage_start", extra={"stage": "peer_review"})
        await emit("review", "Peer review: sending paper to 8 archetype reviewers", None)
        peer_review_result = await run_peer_review_engine(
            api_key=api_key,
            paper_text=paper_text,
            analysis_result=analysis,
            model=models_cfg["archetype"],
            emit=emit,
            tracker=tracker,
            on_call=on_call,
        )
        # Convert reviews and simulations to JSON-serializable dicts
        peer_reviews_dict = reviews_to_dict(peer_review_result.individual_reviews)
        simulations_dict = consolidated_to_dict(peer_review_result)
        # Also attach individual reviews for per-archetype detail
        # (peer_reviews_dict and simulations_dict are passed directly to publish_block)
        logger.info(
            "peer_review_done",
            extra={
                "acceptance_prob": peer_review_result.acceptance_probability,
                "major_concerns": len(peer_review_result.major_concerns),
                "sim_count": len(peer_review_result.recommended_simulations),
            },
        )
        logger.info("stage_end", extra={"stage": "peer_review"})

        # 4. Native HTML5 canvas simulations generated from the reviewed paper.
        logger.info("stage_start", extra={"stage": "simulation_generation"})
        await emit("simulate", "Generating native HTML5 canvas simulations from peer review", None)
        simulations = generate_html5_simulations(
            paper_text=paper_text,
            analysis_result=analysis,
            peer_reviews=peer_reviews_dict,
            recommended_simulations=simulations_dict.get("recommended_simulations", []),
        )
        await emit(
            "simulate_done",
            f"Generated {len(simulations)} runnable HTML5 canvas simulations",
            {"simulation_count": len(simulations)},
        )
        logger.info(
            "stage_end",
            extra={"stage": "simulation_generation", "simulation_count": len(simulations)},
        )

        # 5. Topic derivation
        logger.info("stage_start", extra={"stage": "topic_derivation"})
        await emit("derive", "Deriving next-block topics", None)
        analysis_text_for_derive = json.dumps(
            {"consensus": analysis.consensus_dim, "catalysts": analysis.headline_catalysts},
            indent=2,
        )
        try:
            topics_payload = await chat_json(
                api_key,
                models_cfg["topic_deriver"],
                topic_deriver_prompt(paper_text, analysis_text_for_derive),
                temperature=0.5,
                max_tokens=1500,
                role="topic_deriver",
                tracker=tracker,
                on_call=on_call,
            )
            derived_topics = topics_payload.get("topics", [])
        except Exception as exc:
            logger.warning("topic_derivation_failed", extra={"error": str(exc)})
            await emit("derive", f"topic derivation failed: {exc}", {"error": str(exc)})
            derived_topics = []
        logger.info("stage_end", extra={"stage": "topic_derivation"})

        # 6. Publish
        if auto_publish:
            logger.info("stage_start", extra={"stage": "publish"})
            await emit("publish", "Writing block to results/", None)
            # Build analysis payload including peer reviews & simulations
            publish_meta = publish_block(
                paper=paper,
                analysis=analysis,
                derived_topics=derived_topics,
                research_goal=research_goal,
                models=models_cfg,
                peer_reviews=peer_reviews_dict,
                simulations=simulations,
            )
            logger.info(
                "publish_done",
                extra={"block": publish_meta.get("block"), "path": publish_meta.get("path")},
            )
            await emit(
                "publish_done",
                f"Wrote block {publish_meta['block']} → {publish_meta['path']}",
                publish_meta,
            )

            if git_push:
                push_status = try_git_publish(publish_meta["block"])
                logger.info("git_push", extra={"status": push_status})
                await emit("git", push_status, {"status": push_status})
            logger.info("stage_end", extra={"stage": "publish"})

        stats = tracker.stats()
        logger.info(
            "run_complete",
            extra={
                "title": paper.title,
                "market_price": analysis.market_price,
                "total_calls": stats["total_calls"],
                "total_tokens": stats["total_tokens"],
                "total_cost_usd": stats["total_cost_usd"],
                "elapsed_seconds": stats["elapsed_seconds"],
                "block": publish_meta.get("block") if auto_publish else None,
            },
        )
        await ws.send_text(json.dumps({
            "stage": "done",
            "message": (
                f"✓ Complete · market price = {analysis.market_price:.2f} · "
                f"{stats['total_calls']} API calls · "
                f"{stats['total_tokens']:,} tokens · "
                f"${stats['total_cost_usd']:.4f} · "
                f"{stats['elapsed_seconds']:.0f}s"
            ),
            "data": {
                "title": paper.title,
                "market_price": analysis.market_price,
                "block": publish_meta["block"] if auto_publish else None,
                "result_url": (
                    f"/results/block-{publish_meta['block']}/paper.html"
                    if auto_publish else None
                ),
                "stats": stats,
                "calls": [c.to_dict() for c in tracker.calls],
            },
        }))
    except WebSocketDisconnect:
        logger.warning("client_disconnected")
        return
    except Exception as exc:
        tb = traceback.format_exc()
        logger.error("run_failed", extra={"error_type": type(exc).__name__, "error": str(exc)})
        await emit("error", f"{type(exc).__name__}: {exc}", {"traceback": tb})
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


# Curated OpenRouter model list. Add/remove freely — frontend just reads
# this list and shows it in dropdowns.
MODELS = [
    "anthropic/claude-opus-4.7",
    "anthropic/claude-sonnet-4.6",
    "anthropic/claude-haiku-4.5",
    "openai/gpt-4o",
    "openai/gpt-4o-mini",
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "google/gemini-2.0-flash-001",
    "google/gemini-2.0-flash-lite-001",
    "meta-llama/llama-3.1-70b-instruct",
    "mistralai/mistral-large",
    "deepseek/deepseek-r1",
    "deepseek/deepseek-chat",
    "x-ai/grok-2-1212",
]

DEFAULT_MODELS = {
    "submitter": "openai/gpt-4o-mini",
    "validator": "anthropic/claude-haiku-4.5",
    "compiler": "anthropic/claude-sonnet-4.6",
    "archetype": "anthropic/claude-haiku-4.5",
    "topic_deriver": "anthropic/claude-haiku-4.5",
}


if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")
