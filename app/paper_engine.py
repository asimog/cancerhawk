"""Full MOTO paper engine — adaptive, indefinite aggregation for cancer research.

This is the *full* MOTO stack (not the simplified 3-round / 3-accept variant).
The loop runs until the cancer-research field, under the supplied goal, is
exhausted — detected by saturation and novelty-plateau signals — rather than
by a fixed round count. Every accepted submission accumulates into a shared
research aggregate that is fed back to subsequent submitter rounds, so each
round extends the prior frontier instead of repeating it.

Pipeline:
  1. Round R: spawn N parallel submitters. Each receives the research goal,
     all prior ACCEPTED directions (so it can extend, not duplicate), and the
     last few rejection-steering notes (so it avoids known failure modes).
  2. Validator scores every submission and returns numeric novelty / etc.
  3. Accepted submissions append to the aggregate; rejection feedback is
     pushed back into the next round's submitters as steering.
  4. After each round, the convergence detector decides whether the field
     is exhausted (saturation + novelty plateau) — if not, run another round.
  5. Once converged: compile a single coherent paper from the full aggregate.
  6. (Caller then runs MiroShark peer review over the compiled paper.)

Stop conditions (any one fires, after MIN_ACCEPTED_FLOOR is met):
  - Saturation:        SATURATION_ROUNDS consecutive rounds with 0 accepts.
  - Novelty plateau:   PLATEAU_ROUNDS rounds of non-increasing avg novelty.
  - Soft safety:       MAX_API_CALLS_SOFT or MAX_WALL_CLOCK_SECONDS guards
                       (defaults are deliberately very high — full MOTO is
                       supposed to run as long as the field has signal).

All thresholds are tunable via environment variables; there are no hard
round/accept caps — this is the difference from the simplified variant.
"""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable

from .openrouter import CallEmitFn, chat, chat_json
from .prompts import (
    compiler_outline_prompt,
    compiler_section_prompt,
    submitter_prompt,
    validator_prompt,
)
from .token_tracker import TokenTracker

EmitFn = Callable[[str, str, dict | None], Awaitable[None]]


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


# Adaptive-convergence knobs. None of these are hard caps on rounds or
# acceptances — they're signals the convergence detector consumes.
MIN_ACCEPTED_FLOOR = _env_int("CANCERHAWK_MIN_ACCEPTED", 3)
SATURATION_ROUNDS = _env_int("CANCERHAWK_SATURATION_ROUNDS", 2)
PLATEAU_ROUNDS = _env_int("CANCERHAWK_PLATEAU_ROUNDS", 3)

# Soft safety guards (very high — meant to catch runaway loops, not bound
# normal runs). Set to 0 to disable.
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
    """Return (should_stop, reason).

    Safety guards (api-call and wall-clock) are checked FIRST so a
    pathological run (e.g. every submitter erroring forever) can't loop
    indefinitely just because the acceptance floor was never reached.
    """
    # Hard safety guards — must fire even if MIN_ACCEPTED_FLOOR is unmet.
    if MAX_API_CALLS_SOFT and api_calls >= MAX_API_CALLS_SOFT:
        return True, f"safety_guard:api_calls>={MAX_API_CALLS_SOFT}"
    if MAX_WALL_CLOCK_SECONDS and elapsed_s >= MAX_WALL_CLOCK_SECONDS:
        return True, f"safety_guard:wall_clock>={MAX_WALL_CLOCK_SECONDS}s"

    # Floor: don't stop on convergence signals until we have a real aggregate.
    if accepted_count < MIN_ACCEPTED_FLOOR:
        return False, ""

    # Saturation: K consecutive rounds with zero acceptances.
    if rounds_run >= SATURATION_ROUNDS:
        recent = accepts_per_round[-SATURATION_ROUNDS:]
        if all(c == 0 for c in recent):
            return True, f"saturation:{SATURATION_ROUNDS}_rounds_no_accepts"

    # Novelty plateau: K rounds with non-increasing avg novelty.
    if len(novelty_per_round) >= PLATEAU_ROUNDS:
        window = novelty_per_round[-PLATEAU_ROUNDS:]
        non_increasing = all(window[i] <= window[i - 1] for i in range(1, len(window)))
        if non_increasing:
            return True, f"plateau:novelty_flat_{PLATEAU_ROUNDS}_rounds"

    return False, ""


def _normalize_section_specs(raw_sections: object) -> list[dict[str, str]]:
    """Coerce compiler outline sections into the shape section prompts need."""
    if not isinstance(raw_sections, list):
        return []

    normalized: list[dict[str, str]] = []
    for index, item in enumerate(raw_sections, start=1):
        if isinstance(item, dict):
            heading = str(item.get("heading") or item.get("title") or f"Section {index}").strip()
            summary = str(item.get("summary") or item.get("intent") or item.get("description") or "").strip()
        elif isinstance(item, str):
            heading = item.strip() or f"Section {index}"
            summary = heading
        else:
            continue

        normalized.append({
            "heading": heading or f"Section {index}",
            "summary": summary or "Develop this section from the accepted research aggregate.",
        })

    return normalized


async def run_paper_engine(
    api_key: str,
    research_goal: str,
    models: dict,
    n_submitters: int,
    emit: EmitFn,
    tracker: TokenTracker,
    on_call: CallEmitFn,
    previous_block_context: str = "",
) -> Paper:
    accepted: list[str] = []
    rejection_reasons: list[str] = []
    rejection_log: list[dict] = []

    accepts_per_round: list[int] = []
    novelty_per_round: list[float] = []

    round_num = 0
    started_at = time.time()
    convergence_reason = ""

    await emit(
        "brainstorm",
        "Full MOTO adaptive aggregation: no fixed round cap; "
        f"min_accepted={MIN_ACCEPTED_FLOOR} · saturation={SATURATION_ROUNDS} · "
        f"plateau={PLATEAU_ROUNDS} · soft guards calls<{MAX_API_CALLS_SOFT} "
        f"wall<{MAX_WALL_CLOCK_SECONDS}s",
        {
            "min_accepted": MIN_ACCEPTED_FLOOR,
            "saturation_rounds": SATURATION_ROUNDS,
            "plateau_rounds": PLATEAU_ROUNDS,
            "max_api_calls_soft": MAX_API_CALLS_SOFT,
            "max_wall_clock_seconds": MAX_WALL_CLOCK_SECONDS,
        },
    )

    while True:
        round_num += 1
        await emit(
            "brainstorm",
            f"Round {round_num}: spawning {n_submitters} parallel submitters "
            f"· aggregate size {len(accepted)}",
            {
                "round": round_num,
                "accepted_so_far": len(accepted),
                "elapsed_s": round(time.time() - started_at, 1),
            },
        )

        # Parallel submissions — each submitter sees the full prior aggregate
        # so it extends rather than duplicates (full MOTO aggregation).
        submissions = await asyncio.gather(
            *[
                chat(
                    api_key,
                    models["submitter"],
                    submitter_prompt(
                        research_goal,
                        rejection_reasons,
                        prior_accepted=accepted,
                        previous_block_context=previous_block_context,
                    ),
                    temperature=0.85,
                    role="submitter",
                    tracker=tracker,
                    on_call=on_call,
                )
                for _ in range(n_submitters)
            ],
            return_exceptions=True,
        )

        round_accepts = 0
        round_novelty_scores: list[float] = []

        for i, sub in enumerate(submissions):
            if isinstance(sub, Exception):
                await emit("validate", f"Submitter {i + 1} failed: {sub}", {"error": str(sub)})
                continue
            verdict = await chat_json(
                api_key,
                models["validator"],
                validator_prompt(sub),
                temperature=0.3,
                role="validator",
                tracker=tracker,
                on_call=on_call,
            )
            # LLMs sometimes wrap the response in a list: [{...}] instead of {...}
            if isinstance(verdict, list) and verdict:
                verdict = verdict[0]
            if not isinstance(verdict, dict):
                await emit("validate", f"✗ validator returned unexpected type: {type(verdict).__name__}", {"verdict": str(verdict)[:200]})
                continue
            scores = verdict.get("scores") or {}
            nov = scores.get("novelty")
            if isinstance(nov, (int, float)):
                round_novelty_scores.append(float(nov))

            if verdict.get("accept"):
                accepted.append(sub)
                round_accepts += 1
                await emit(
                    "validate",
                    f"✓ accepted submission {len(accepted)} "
                    f"(round {round_num}): {verdict.get('reason', '')[:120]}",
                    {"scores": scores, "accepted_total": len(accepted)},
                )
            else:
                steering = verdict.get("steering_feedback") or verdict.get("reason") or ""
                rejection_reasons.append(steering[:200])
                rejection_log.append({"submission": sub[:300], "feedback": steering[:300]})
                await emit(
                    "validate",
                    f"✗ rejected — {steering[:120]}",
                    {"scores": scores},
                )

        accepts_per_round.append(round_accepts)
        round_avg_novelty = (
            sum(round_novelty_scores) / len(round_novelty_scores)
            if round_novelty_scores else 0.0
        )
        novelty_per_round.append(round_avg_novelty)

        api_calls = len(tracker.calls) if hasattr(tracker, "calls") else 0
        elapsed_s = time.time() - started_at

        await emit(
            "brainstorm",
            f"Round {round_num} closed · +{round_accepts} accepted · "
            f"avg_novelty={round_avg_novelty:.1f} · total_accepted={len(accepted)}",
            {
                "round": round_num,
                "round_accepts": round_accepts,
                "round_avg_novelty": round_avg_novelty,
                "total_accepted": len(accepted),
                "api_calls": api_calls,
                "elapsed_s": round(elapsed_s, 1),
            },
        )

        should_stop, reason = _check_convergence(
            accepted_count=len(accepted),
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
                f"Converged after {round_num} rounds · {len(accepted)} accepted · "
                f"reason={reason}",
                {
                    "rounds": round_num,
                    "accepted": len(accepted),
                    "reason": reason,
                    "accepts_per_round": accepts_per_round,
                    "novelty_per_round": novelty_per_round,
                },
            )
            break

    if not accepted:
        raise RuntimeError("No submissions were accepted across the adaptive run")

     # Compile outline.
     await emit("compile", "Compiling paper outline from full research aggregate", None)
     outline = await chat_json(
         api_key,
         models["compiler"],
         compiler_outline_prompt(accepted, research_goal, previous_block_context),
         temperature=0.4,
         max_tokens=4000,
         role="compiler_outline",
         tracker=tracker,
         on_call=on_call,
     )
     # LLMs sometimes wrap responses in arrays
     if isinstance(outline, list) and outline:
         outline = outline[0]
     if not isinstance(outline, dict):
         raise RuntimeError(f"Compiler returned unexpected type: {type(outline).__name__}")
     title = outline.get("title", "Untitled CancerHawk Paper")
    section_specs = _normalize_section_specs(outline.get("sections"))
    if not section_specs:
        raise RuntimeError("Compiler returned no sections")

    await emit(
        "compile",
        f"Outline ready: '{title}' · {len(section_specs)} sections",
        {"title": title, "section_count": len(section_specs)},
    )

    # Write sections sequentially.
    written: list[dict] = []
    for i, spec in enumerate(section_specs):
        await emit(
            "compile",
            f"Writing section {i + 1}/{len(section_specs)}: {spec.get('heading', '')}",
            {"section_index": i, "heading": spec.get("heading", "")},
        )
        content = await chat(
            api_key,
            models["compiler"],
            compiler_section_prompt(title, spec, written, research_goal, previous_block_context),
            temperature=0.55,
            max_tokens=2200,
            role="compiler_section",
            tracker=tracker,
            on_call=on_call,
        )
        written.append({"heading": spec.get("heading", f"Section {i + 1}"), "content": content.strip()})

    return Paper(
        title=title,
        sections=written,
        accepted_submissions=accepted,
        rejections=rejection_log,
        rounds_run=round_num,
        convergence_reason=convergence_reason,
    )
