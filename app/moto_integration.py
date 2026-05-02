"""MOTO integration for CancerHawk.

Replaces the original paper_engine with the upstream MOTO aggregator + compiler.
Provides the same interface: run_moto_engine(api_key, research_goal, models, n_submitters, emit, tracker, on_call, previous_block_context)
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable

from .openrouter import CallEmitFn, chat, chat_json
from .token_tracker import TokenTracker

from backend.shared.config import rag_config, system_config
from backend.shared.api_client_manager import api_client_manager
from backend.aggregator.core.coordinator import coordinator
from backend.aggregator.memory.shared_training import shared_training_memory
from backend.compiler.core.compiler_coordinator import compiler_coordinator
from backend.compiler.memory.paper_memory import paper_memory
from backend.compiler.memory.outline_memory import outline_memory
from backend.compiler.core.compiler_rag_manager import compiler_rag_manager
from backend.shared.lm_studio_client import lm_studio_client

logger = logging.getLogger("cancerhawk.moto")

EmitFn = Callable[[str, str, dict | None], Awaitable[None]]


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


# Convergence knobs (matching original CancerHawk behavior)
MIN_ACCEPTED_FLOOR = _env_int("CANCERHAWK_MIN_ACCEPTED", 3)
SATURATION_ROUNDS = _env_int("CANCERHAWK_SATURATION_ROUNDS", 2)
PLATEAU_ROUNDS = _env_int("CANCERHAWK_PLATEAU_ROUNDS", 3)
MAX_API_CALLS_SOFT = _env_int("CANCERHAWK_MAX_CALLS", 400)
MAX_WALL_CLOCK_SECONDS = _env_int("CANCERHAWK_MAX_WALL_CLOCK", 3600)


@dataclass
class Paper:
    title: str
    sections: list[dict]  # [{"heading": str, "content": str}]
    accepted_submissions: list[str]
    rejections: list[dict] = field(default_factory=list)
    rounds_run: int = 0
    convergence_reason: str = ""

    def full_text(self) -> str:
        body = "\n\n".join(f"## {s['heading']}\n\n{s['content']}" for s in self.sections)
        return f"# {self.title}\n\n{body}"


def _check_convergence(
    *,
    accepted_count: int,
    rounds_run: int,
    accepts_per_round: list[int],
    novelty_per_round: list[float],
    api_calls: int,
    elapsed_s: float,
) -> tuple[bool, str]:
    """MOTO convergence detection (adaptive, not fixed round count)."""
    if MAX_API_CALLS_SOFT and api_calls >= MAX_API_CALLS_SOFT:
        return True, f"safety_guard:api_calls>={MAX_API_CALLS_SOFT}"
    if MAX_WALL_CLOCK_SECONDS and elapsed_s >= MAX_WALL_CLOCK_SECONDS:
        return True, f"safety_guard:wall_clock>={MAX_WALL_CLOCK_SECONDS}s"

    if accepted_count < MIN_ACCEPTED_FLOOR:
        return False, ""

    if rounds_run >= SATURATION_ROUNDS:
        recent = accepts_per_round[-SATURATION_ROUNDS:]
        if all(c == 0 for c in recent):
            return True, f"saturation:{SATURATION_ROUNDS}_rounds_no_accepts"

    if len(novelty_per_round) >= PLATEAU_ROUNDS:
        window = novelty_per_round[-PLATEAU_ROUNDS:]
        non_increasing = all(window[i] <= window[i - 1] for i in range(1, len(window)))
        if non_increasing:
            return True, f"plateau:novelty_flat_{PLATEAU_ROUNDS}_rounds"

    return False, ""


async def _initialize_moto_components(api_key: str, models: dict, n_submitters: int):
    """Initialize MOTO aggregator and compiler coordinator with CancerHawk prompts."""
    from backend.aggregator.prompts.submitter_prompts import build_submitter_prompt
    from backend.aggregator.prompts.validator_prompts import build_validator_prompt

    # Configure API key
    if api_key:
        api_client_manager.set_openrouter_api_key(api_key)

    # Initialize trainers
    rag_config.validator_context_window = 131072
    rag_config.validator_max_output_tokens = 25000
    rag_config.submitter_context_window = 131072
    rag_config.submitter_max_output_tokens = 25000

    # Configure roles in API client manager
    from backend.shared.models import ModelConfig, SubmitterConfig

    submitter_configs = []
    for i in range(1, n_submitters + 1):
        config = SubmitterConfig(
            submitter_id=i,
            provider="lm_studio",
            model_id=models.get("submitter", "openrouter/free"),
            context_window=131072,
            max_output_tokens=25000
        )
        submitter_configs.append(config)

    # Initialize compiler RAG manager
    await compiler_rag_manager.initialize()

    return submitter_configs


async def run_moto_engine(
    api_key: str,
    research_goal: str,
    models: dict,
    n_submitters: int,
    emit: EmitFn,
    tracker: TokenTracker,
    on_call: CallEmitFn,
    previous_block_context: str = "",
) -> Paper:
    """
    Run MOTO's adaptive aggregator + compiler pipeline on an oncology research goal.
    This replaces the original paper_engine with upstream MOTO architecture.
    """
    logger.info("Starting MOTO integration (upstream v1.0.7 architecture)")

    # Initialize MOTO components
    submitter_configs = await _initialize_moto_components(api_key, models, n_submitters)

    # ===== PHASE 1: AGGREGATOR (Adaptive brainstorming) =====
    await emit("brainstorm", "MOTO aggregator: starting adaptive brainstorming", None)

    accepted_submissions = []
    rejection_reasons = []
    accepts_per_round = []
    novelty_per_round = []
    round_num = 0
    started_at = time.time()
    convergence_reason = ""

    while True:
        round_num += 1
        await emit(
            "brainstorm",
            f"Round {round_num}: spawning {n_submitters} parallel submitters · aggregate size {len(accepted_submissions)}",
            {"round": round_num, "accepted_so_far": len(accepted_submissions)},
        )

        # For MOTO integration, we'll use a simplified parallel submit loop
        # In full upstream MOTO, this uses coordinator + queue system
        # Here we directly call chat() to match CancerHawk's current token tracking pattern

        round_accepts = 0
        round_novelty_scores = []

        # Generate submissions in parallel (like MOTO submitters)
        tasks = []
        for i in range(n_submitters):
            task = _generate_moto_submission(
                api_key=api_key,
                model=models["submitter"],
                research_goal=research_goal,
                prior_accepted=accepted_submissions,
                prior_rejections=rejection_reasons[-5:],
                previous_block_context=previous_block_context,
                tracker=tracker,
                on_call=on_call,
                submitter_id=i + 1,
            )
            tasks.append(task)

        submissions = await asyncio.gather(*tasks, return_exceptions=True)

        for i, sub in enumerate(submissions):
            if isinstance(sub, Exception):
                await emit("validate", f"Submitter {i + 1} failed: {sub}", {"error": str(sub)})
                continue

            # Validate with MOTO validator (batch validation would be used in full upstream version)
            verdict = await _validate_moto_submission(
                api_key=api_key,
                model=models["validator"],
                submission=sub,
                research_goal=research_goal,
                tracker=tracker,
                on_call=on_call,
            )

            nov = verdict.get("scores", {}).get("novelty")
            if isinstance(nov, (int, float)):
                round_novelty_scores.append(float(nov))

            if verdict.get("accept"):
                accepted_submissions.append(sub)
                round_accepts += 1
                await emit(
                    "validate",
                    f"✓ accepted submission {len(accepted_submissions)} (round {round_num}): {verdict.get('reason', '')[:120]}",
                    {"scores": verdict.get("scores", {})},
                )
            else:
                steering = verdict.get("steering_feedback") or verdict.get("reason") or ""
                rejection_reasons.append(steering[:200])
                await emit(
                    "validate",
                    f"✗ rejected — {steering[:120]}",
                    {"scores": verdict.get("scores", {})},
                )

        accepts_per_round.append(round_accepts)
        round_avg_novelty = sum(round_novelty_scores) / len(round_novelty_scores) if round_novelty_scores else 0.0
        novelty_per_round.append(round_avg_novelty)

        api_calls = len(tracker.calls) if hasattr(tracker, "calls") else 0
        elapsed_s = time.time() - started_at

        await emit(
            "brainstorm",
            f"Round {round_num} closed · +{round_accepts} accepted · avg_novelty={round_avg_novelty:.1f} · total_accepted={len(accepted_submissions)}",
            {
                "round": round_num,
                "round_accepts": round_accepts,
                "round_avg_novelty": round_avg_novelty,
                "total_accepted": len(accepted_submissions),
                "api_calls": api_calls,
                "elapsed_s": round(elapsed_s, 1),
            },
        )

        should_stop, reason = _check_convergence(
            accepted_count=len(accepted_submissions),
            rounds_run=round_num,
            accepts_per_round=accepts_per_round,
            novelty_per_round=novelty_per_round,
            api_calls=api_calls,
            elapsed_s=elapsed_s,
        )
        if should_stop:
            convergence_reason = reason
            await emit(
                "brainstorm",
                f"Converged after {round_num} rounds · {len(accepted_submissions)} accepted · reason={reason}",
                {
                    "rounds": round_num,
                    "accepted": len(accepted_submissions),
                    "reason": reason,
                    "accepts_per_round": accepts_per_round,
                    "novelty_per_round": novelty_per_round,
                },
            )
            break

    if not accepted_submissions:
        raise RuntimeError("No submissions were accepted across the adaptive run")

    # ===== PHASE 2: COMPILER (Paper generation) =====
    await emit("compile", "MOTO compiler: generating paper from aggregate", None)

    # Use MOTO compiler prompts (these are the upstream v1.0.7 prompts)
    from backend.compiler.prompts.outline_prompts import build_outline_prompt
    from backend.compiler.prompts.construction_prompts import get_body_construction_system_prompt, build_construction_prompt
    from backend.compiler.prompts.construction_prompts import get_conclusion_construction_system_prompt, build_conclusion_prompt
    from backend.compiler.prompts.final_answer_prompts import INTRO_OUTLINE_PROMPT, build_intro_prompt, build_abstract_prompt

    # Step A: Create outline
    outline_prompt = build_outline_prompt(accepted_submissions, research_goal, previous_block_context)
    outline_resp = await chat(
        api_key,
        models["compiler"],
        [{"role": "system", "content": get_body_construction_system_prompt()}, {"role": "user", "content": outline_prompt}],
        temperature=0.4,
        max_tokens=4000,
        role="compiler_outline",
        tracker=tracker,
        on_call=on_call,
    )

    # Parse outline (MOTO uses stricter format - extract JSON)
    import json
    try:
        outline_data = json.loads(outline_resp)
        title = outline_data.get("title", "Untitled CancerHawk Paper")
        sections_spec = outline_data.get("sections", [])
    except json.JSONDecodeError:
        # Fallback: treat as plain text and extract title/sections heuristically
        title = "Untitled CancerHawk Paper"
        sections_spec = [{"heading": "Introduction", "summary": "Standard introduction"}]

    await emit(
        "compile",
        f"Outline ready: '{title}' · {len(sections_spec)} sections",
        {"title": title, "section_count": len(sections_spec)},
    )

    # Step B: Write body sections (construction mode)
    written_sections = []
    for i, spec in enumerate(sections_spec):
        heading = spec.get("heading", f"Section {i + 1}")
        await emit("compile", f"Writing section {i + 1}/{len(sections_spec)}: {heading}", {"section_index": i, "heading": heading})

        # Build construction prompt
        prior_sections_text = "\n".join(f"### {s['heading']}\n{s['content'][:500]}" for s in written_sections)
        construction_prompt = build_construction_prompt(
            outline_spec=spec,
            prior_sections=prior_sections_text,
            research_goal=research_goal,
            previous_block_context=previous_block_context,
        )

        section_content = await chat(
            api_key,
            models["compiler"],
            [{"role": "system", "content": get_body_construction_system_prompt()}, {"role": "user", "content": construction_prompt}],
            temperature=0.55,
            max_tokens=2200,
            role="compiler_section",
            tracker=tracker,
            on_call=on_call,
        )
        written_sections.append({"heading": heading, "content": section_content.strip()})

    # Step C: Write conclusion
    await emit("compile", "Writing conclusion section", None)
    conclusion_prompt = build_conclusion_prompt(
        paper_sections=written_sections,
        research_goal=research_goal,
        previous_block_context=previous_block_context,
    )
    conclusion_content = await chat(
        api_key,
        models["compiler"],
        [{"role": "system", "content": get_conclusion_construction_system_prompt()}, {"role": "user", "content": conclusion_prompt}],
        temperature=0.6,
        max_tokens=1500,
        role="compiler_conclusion",
        tracker=tracker,
        on_call=on_call,
    )
    written_sections.append({"heading": "Conclusion", "content": conclusion_content.strip()})

    # Step D: Write introduction (after body+conclusion)
    await emit("compile", "Writing introduction section", None)
    intro_outline_prompt = INTRO_OUTLINE_PROMPT.format(research_goal=research_goal)
    intro_prompt = build_intro_prompt(
        paper_sections=written_sections,
        intro_outline=intro_outline_prompt,
        previous_block_context=previous_block_context,
    )
    intro_content = await chat(
        api_key,
        models["compiler"],
        [{"role": "system", "content": get_body_construction_system_prompt()}, {"role": "user", "content": intro_prompt}],
        temperature=0.5,
        max_tokens=1500,
        role="compiler_intro",
        tracker=tracker,
        on_call=on_call,
    )
    # Insert introduction as second section (after title/abstract position)
    final_sections = [{"heading": "Introduction", "content": intro_content.strip()}]
    # Remove placeholder introduction if present and add real one
    for s in written_sections:
        if s["heading"].lower() != "introduction" and s["heading"].lower() != "conclusion":
            final_sections.append(s)
    final_sections.append({"heading": "Conclusion", "content": conclusion_content.strip()})

    # Step E: Write abstract
    await emit("compile", "Writing abstract", None)
    abstract_prompt = build_abstract_prompt(
        paper_sections=final_sections,
        research_goal=research_goal,
        previous_block_context=previous_block_context,
    )
    abstract_content = await chat(
        api_key,
        models["compiler"],
        [{"role": "system", "content": get_body_construction_system_prompt()}, {"role": "user", "content": abstract_prompt}],
        temperature=0.4,
        max_tokens=500,
        role="compiler_abstract",
        tracker=tracker,
        on_call=on_call,
    )
    final_sections.insert(0, {"heading": "Abstract", "content": abstract_content.strip()})

    # Build final paper
    paper = Paper(
        title=title,
        sections=final_sections,
        accepted_submissions=accepted_submissions,
        rejections=[],  # Could populate from rejection log if needed
        rounds_run=round_num,
        convergence_reason=convergence_reason,
    )

    await emit("compile", f"Paper complete: '{title}' · {len(final_sections)} sections", {
        "title": title,
        "section_count": len(final_sections),
        "total_words": len(paper.full_text().split()),
    })

    return paper


# Helper functions for simplified MOTO submission/validation (without full coordinator)
async def _generate_moto_submission(
    api_key: str,
    model: str,
    research_goal: str,
    prior_accepted: list[str],
    prior_rejections: list[str],
    previous_block_context: str,
    tracker: TokenTracker,
    on_call: CallEmitFn,
    submitter_id: int,
) -> str:
    """Generate a single submission using MOTO submitter prompt."""
    from backend.aggregator.prompts.submitter_prompts import build_submitter_prompt

    rejections = ""
    if prior_rejections:
        rejections = "\nPRIOR REJECTIONS (do not repeat these failure modes):\n- " + "\n- ".join(
            prior_rejections[-5:]
        )

    aggregate = ""
    if prior_accepted:
        snippets = []
        for i, sub in enumerate(prior_accepted, start=1):
            stripped = (sub or "").strip()
            lines = stripped.splitlines() if stripped else []
            head = lines[0][:200] if lines else "(empty submission)"
            snippets.append(f"  [{i}] {head}")
        aggregate = (
            "\nALREADY-ACCEPTED RESEARCH DIRECTIONS (do NOT duplicate; extend the frontier):\n"
            + "\n".join(snippets)
            + "\n\nYour submission must add a *new* mechanism, target, modality, "
            "or patient context that the aggregate above does not already cover. "
            "Adjacent extensions are welcome; near-duplicates will be rejected."
        )

    prior_blocks = ""
    if previous_block_context.strip():
        prior_blocks = (
            "\nPREVIOUS CANCERHAWK BLOCKS YOU MAY CITE OR EXTEND WHEN RELEVANT:\n"
            f"{previous_block_context[:6000]}\n\n"
            "If a prior block is relevant, explicitly cite it as "
            "`CancerHawk Block N` and explain whether you are extending, "
            "challenging, or reusing its mechanism. Do not repeat prior work."
        )

    prompt = build_submitter_prompt(
        user_prompt=research_goal,
        context=f"{aggregate}\n{prior_blocks}\n{rejections}",
        rag_evidence="",
    )

    response = await chat(
        api_key,
        model,
        prompt,
        temperature=0.85,
        role=f"submitter_{submitter_id}",
        tracker=tracker,
        on_call=on_call,
        max_tokens=25000,
    )

    # Extract submission content (MOTO returns JSON with "submission" and "reasoning")
    try:
        import json
        parsed = json.loads(response)
        return parsed.get("submission", response)
    except json.JSONDecodeError:
        # If not JSON, return full response
        return response


async def _validate_moto_submission(
    api_key: str,
    model: str,
    submission: str,
    research_goal: str,
    tracker: TokenTracker,
    on_call: CallEmitFn,
) -> dict:
    """Validate a submission using MOTO validator prompt."""
    from backend.aggregator.prompts.validator_prompts import build_validator_prompt

    prompt = build_validator_prompt(
        user_prompt=research_goal,
        submission_content=submission,
        context="",
        rag_evidence="",
    )

    response = await chat_json(
        api_key,
        model,
        prompt,
        temperature=0.3,
        role="validator",
        tracker=tracker,
        on_call=on_call,
        max_tokens=2000,
    )

    if isinstance(response, list) and response:
        response = response[0]
    if not isinstance(response, dict):
        return {"accept": False, "reasoning": "Validator returned invalid format", "scores": {}}

    return response
