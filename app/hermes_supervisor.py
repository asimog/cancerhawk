"""Hermes supervisor for CancerHawk runs.

This module is the Railway-side orchestrator: it owns the run lifecycle,
executes the full MOTO -> analysis -> peer review -> simulation -> publish
pipeline, and hands completed artifacts to the GitHub publisher so Vercel can
rebuild the website.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from .analysis_engine import run_analysis_engine
from .openrouter import chat_json
from .paper_engine import run_paper_engine
from .peer_review_engine import (
    consolidated_to_dict,
    reviews_to_dict,
    run_peer_review_engine,
)
from .prompts import topic_deriver_prompt
from .publisher import hydrate_results_from_github, load_previous_block_context, publish_block, try_git_publish, stage_block
from .simulation_engine import generate_html5_simulations
from .token_tracker import APICall, TokenTracker

logger = logging.getLogger("cancerhawk.hermes")

EmitFn = Callable[[str, str, dict | None], Awaitable[None]]
CallEmitFn = Callable[[APICall], Awaitable[None]]


@dataclass
class HermesRunConfig:
    api_key: str
    research_goal: str
    n_submitters: int
    auto_publish: bool
    git_push: bool
    models: dict[str, str]
    job_id: str | None = None
    stage: bool = False


@dataclass
class HermesRunResult:
    title: str
    market_price: float
    block: int | None
    result_url: str | None
    stats: dict[str, Any]
    calls: list[dict[str, Any]]
    git_status: str | None = None


class HermesSupervisor:
    """Top-level CancerHawk agent that supervises the research run."""

    def __init__(self, *, emit: EmitFn, on_call: CallEmitFn, tracker: TokenTracker | None = None) -> None:
        self.emit = emit
        self.on_call = on_call
        self.tracker = tracker or TokenTracker()

    async def run(self, cfg: HermesRunConfig) -> HermesRunResult:
        tracker = self.tracker
        publish_meta: dict[str, Any] | None = None
        git_status: str | None = None

        await self.emit(
            "hermes",
            "Hermes supervisor started: overseeing MOTO, peer review, simulations, repo publish",
            {
                "auto_publish": cfg.auto_publish,
                "git_push": cfg.git_push,
                "models": cfg.models,
            },
        )

        hydration_status = hydrate_results_from_github()
        await self.emit("hermes", hydration_status, {"hydration_status": hydration_status})

        previous_block_context = load_previous_block_context()
        if previous_block_context:
            await self.emit(
                "prior_blocks",
                "Hermes loaded prior CancerHawk blocks for continuity",
                {"chars": len(previous_block_context)},
            )

        logger.info("stage_start", extra={"stage": "paper_engine", "supervisor": "hermes"})
        paper = await run_paper_engine(
            api_key=cfg.api_key,
            research_goal=cfg.research_goal,
            models=cfg.models,
            n_submitters=cfg.n_submitters,
            emit=self.emit,
            tracker=tracker,
            on_call=self.on_call,
            previous_block_context=previous_block_context,
        )
        logger.info("stage_end", extra={"stage": "paper_engine", "supervisor": "hermes"})
        paper_text = paper.full_text()
        await self.emit(
            "paper_done",
            f"Paper compiled: '{paper.title}' · {len(paper.sections)} sections · "
            f"{len(paper_text.split())} words · aggregated "
            f"{len(paper.accepted_submissions)} directions over "
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

        logger.info("stage_start", extra={"stage": "analysis_engine", "supervisor": "hermes"})
        analysis = await run_analysis_engine(
            api_key=cfg.api_key,
            paper_text=paper_text,
            archetype_model=cfg.models["archetype"],
            emit=self.emit,
            tracker=tracker,
            on_call=self.on_call,
        )
        logger.info("stage_end", extra={"stage": "analysis_engine", "supervisor": "hermes"})

        logger.info("stage_start", extra={"stage": "peer_review", "supervisor": "hermes"})
        await self.emit("review", "Hermes sending paper to MiroShark peer reviewers", None)
        peer_review_result = await run_peer_review_engine(
            api_key=cfg.api_key,
            paper_text=paper_text,
            analysis_result=analysis,
            model=cfg.models["archetype"],
            emit=self.emit,
            tracker=tracker,
            on_call=self.on_call,
        )
        peer_reviews_dict = reviews_to_dict(peer_review_result.individual_reviews)
        simulations_dict = consolidated_to_dict(peer_review_result)
        await self.emit(
            "review_complete",
            f"Hermes peer review complete · acceptance={peer_review_result.acceptance_probability:.0%} · "
            f"{len(peer_review_result.recommended_simulations)} simulation proposals",
            {
                "acceptance_probability": peer_review_result.acceptance_probability,
                "major_concerns": len(peer_review_result.major_concerns),
                "simulation_count": len(peer_review_result.recommended_simulations),
            },
        )
        logger.info("stage_end", extra={"stage": "peer_review", "supervisor": "hermes"})

        logger.info("stage_start", extra={"stage": "simulation_generation", "supervisor": "hermes"})
        await self.emit("simulate", "Hermes generating browser-native simulations from peer review", None)
        simulations = generate_html5_simulations(
            paper_text=paper_text,
            analysis_result=analysis,
            peer_reviews=peer_reviews_dict,
            recommended_simulations=simulations_dict.get("recommended_simulations", []),
        )
        await self.emit(
            "simulate_done",
            f"Hermes generated {len(simulations)} runnable simulations",
            {"simulation_count": len(simulations)},
        )
        logger.info("stage_end", extra={"stage": "simulation_generation", "supervisor": "hermes"})

        logger.info("stage_start", extra={"stage": "topic_derivation", "supervisor": "hermes"})
        await self.emit("derive", "Hermes deriving next-block topics", None)
        analysis_text_for_derive = json.dumps(
            {"consensus": analysis.consensus_dim, "catalysts": analysis.headline_catalysts},
            indent=2,
        )
        try:
            topics_payload = await chat_json(
                cfg.api_key,
                cfg.models["topic_deriver"],
                topic_deriver_prompt(paper_text, analysis_text_for_derive),
                temperature=0.5,
                max_tokens=1500,
                role="topic_deriver",
                tracker=tracker,
                on_call=self.on_call,
            )
            derived_topics = topics_payload.get("topics", [])
        except Exception as exc:
            logger.warning("topic_derivation_failed", extra={"error": str(exc)})
            await self.emit("derive", f"topic derivation failed: {exc}", {"error": str(exc)})
            derived_topics = []
        logger.info("stage_end", extra={"stage": "topic_derivation", "supervisor": "hermes"})

        publish_meta = None
        if getattr(cfg, 'stage', False) and cfg.job_id:
            logger.info("stage_start", extra={"stage": "stage", "supervisor": "hermes"})
            await self.emit("stage", "Hermes staging block artifacts for later publication", {"job_id": cfg.job_id})
            try:
                publish_meta = stage_block(
                    paper=paper,
                    analysis=analysis,
                    derived_topics=derived_topics,
                    research_goal=cfg.research_goal,
                    models=cfg.models,
                    peer_reviews=peer_reviews_dict,
                    simulations=simulations,
                    job_id=cfg.job_id,
                    git_push=cfg.git_push,
                )
            except Exception as exc:
                logger.error("staging_failed", extra={"job_id": cfg.job_id, "error": str(exc)})
                raise
            await self.emit("stage_done", f"Hermes staged artifacts for job {cfg.job_id}", publish_meta)
            logger.info("stage_end", extra={"stage": "stage", "supervisor": "hermes"})
        elif cfg.auto_publish:
            logger.info("stage_start", extra={"stage": "publish", "supervisor": "hermes"})
            await self.emit("publish", "Hermes writing block bundle to results/", None)
            publish_meta = publish_block(
                paper=paper,
                analysis=analysis,
                derived_topics=derived_topics,
                research_goal=cfg.research_goal,
                models=cfg.models,
                peer_reviews=peer_reviews_dict,
                simulations=simulations,
            )
            await self.emit(
                "publish_done",
                f"Hermes wrote block {publish_meta['block']} -> {publish_meta['path']}",
                publish_meta,
            )
            if cfg.git_push:
                await self.emit("git", "Hermes checking out GitHub repo and preparing commit", None)
                git_status = try_git_publish(publish_meta["block"])
                await self.emit("git", git_status, {"status": git_status})
            logger.info("stage_end", extra={"stage": "publish", "supervisor": "hermes"})

        stats = tracker.stats()
        return HermesRunResult(
            title=paper.title,
            market_price=analysis.market_price,
            block=publish_meta.get("block") if publish_meta else None,
            result_url=(f"/results/block-{publish_meta['block']}/paper.html" if publish_meta else None),
            stats=stats,
            calls=[c.to_dict() for c in tracker.calls],
            git_status=git_status,
        )
