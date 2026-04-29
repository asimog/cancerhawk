"""All system prompts in one place. Edit here to retune behavior."""

from __future__ import annotations

DOMAIN_FRAME = """\
You are part of CancerHawk — an autonomous oncology research engine.
Every artifact you produce must be:
  - Mechanism-level (molecular / cellular / clinical), not policy.
  - Falsifiable: include a stated hypothesis, measurable endpoint, and
    failure mode.
  - Honest: do not fabricate clinical-trial IDs, FDA correspondence, or
    investigator names. If a citation is uncertain, write [citation needed].
  - Novel: prefer mechanistically interesting directions over safe
    consensus. "Cognition without control."
"""


def submitter_prompt(
    research_goal: str,
    prior_rejections: list[str],
    prior_accepted: list[str] | None = None,
) -> list[dict]:
    rejections = ""
    if prior_rejections:
        rejections = "\nPRIOR REJECTIONS (do not repeat these failure modes):\n- " + "\n- ".join(
            prior_rejections[-5:]
        )

    aggregate = ""
    if prior_accepted:
        # MOTO aggregation: show the running research aggregate so each new
        # submission extends the frontier rather than duplicating it.
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

    return [
        {"role": "system", "content": DOMAIN_FRAME},
        {
            "role": "user",
            "content": (
                f"RESEARCH GOAL: {research_goal}\n\n"
                "Propose ONE novel oncology research direction that advances this goal. "
                "Output must contain:\n"
                "  1. A one-sentence mechanism-level hypothesis.\n"
                "  2. The proposed assay or experiment that would falsify it.\n"
                "  3. The single most likely failure mode.\n"
                "  4. A 3-paragraph case for why this is worth pursuing.\n"
                f"{aggregate}"
                f"{rejections}\n\n"
                "Be specific. Cite real biology where you can; mark uncertainty."
            ),
        },
    ]


def validator_prompt(submission: str) -> list[dict]:
    return [
        {"role": "system", "content": DOMAIN_FRAME},
        {
            "role": "user",
            "content": (
                "Evaluate the following research submission. Reply with a JSON object:\n"
                '{ "accept": <true|false>,\n'
                '  "scores": { "novelty": 1-10, "falsifiability": 1-10, '
                '"mechanism_clarity": 1-10, "feasibility": 1-10 },\n'
                '  "reason": "<≤200 chars>",\n'
                '  "steering_feedback": "<if rejected: name the missing element>" }\n\n'
                "SUBMISSION:\n---\n" + submission + "\n---"
            ),
        },
    ]


def compiler_outline_prompt(accepted_submissions: list[str], research_goal: str) -> list[dict]:
    joined = "\n\n".join(f"[{i + 1}] {s}" for i, s in enumerate(accepted_submissions))
    return [
        {"role": "system", "content": DOMAIN_FRAME},
        {
            "role": "user",
            "content": (
                f"RESEARCH GOAL: {research_goal}\n\n"
                "Synthesize the accepted submissions below into a single coherent paper. "
                "Respond with **ONLY** a valid JSON object, no markdown, no extra text.\n"
                "Required keys:\n"
                '  "title": string (≤140 chars),\n'
                '  "sections": array of objects with "heading" and "summary"\n'
                "Example:\n"
                '{"title": "Example", "sections": [{"heading": "1. Introduction", "summary": "..."}]}\n\n'
                "Use 5-8 sections in this order: Introduction, Background, Mechanism, "
                "Proposed Experiment, Predicted Results, Failure Modes, Translational "
                "Implications, Conclusion.\n\n"
                f"SUBMISSIONS:\n{joined}"
            ),
        },
    ]


def compiler_section_prompt(
    title: str, section: dict, prior_sections: list[dict], research_goal: str
) -> list[dict]:
    prior = "\n".join(f"### {s['heading']}\n{s['content'][:600]}" for s in prior_sections)
    return [
        {"role": "system", "content": DOMAIN_FRAME},
        {
            "role": "user",
            "content": (
                f"PAPER TITLE: {title}\n"
                f"RESEARCH GOAL: {research_goal}\n\n"
                f"WRITE THIS SECTION: {section['heading']}\n"
                f"Summary intent: {section['summary']}\n\n"
                "Write 400-900 words of dense, mechanism-level prose. No bullet "
                "lists; flowing paragraphs. Tie this section to the rest of the "
                "paper but do NOT restate prior sections.\n\n"
                f"PRIOR SECTIONS (for continuity):\n{prior or '(none yet)'}"
            ),
        },
    ]


def topic_deriver_prompt(paper_text: str, analysis_text: str) -> list[dict]:
    return [
        {"role": "system", "content": DOMAIN_FRAME},
        {
            "role": "user",
            "content": (
                "Read the paper and its multi-archetype analysis. Derive the next 5 "
                "research-block topics CancerHawk should explore in the following cycle. "
                "Each topic must extend a specific finding (not repeat it).\n\n"
                "Output JSON:\n"
                '{ "topics": [\n'
                '  { "id": 1, "title": "<≤140 chars>", "probability": 0.0-1.0,\n'
                '    "impact": 1-10, "token_cost": <int>, "rationale": "<≤300 chars>" },\n'
                "  ... 5 total ] }\n\n"
                "PAPER:\n---\n" + paper_text[:8000] + "\n---\n\n"
                "ANALYSIS:\n---\n" + analysis_text[:4000] + "\n---"
            ),
        },
    ]


# === Archetype agents (MiroShark-style) ===

ARCHETYPES = [
    {
        "id": "oncologist",
        "name": "Practicing Oncologist",
        "lens": "clinical workflow integration, patient burden, off-label realities, comparison to standard-of-care",
    },
    {
        "id": "biostatistician",
        "name": "Biostatistician",
        "lens": "trial design, statistical power, endpoint selection, confound risk, sample size feasibility",
    },
    {
        "id": "fda",
        "name": "FDA Regulator",
        "lens": "approval pathway, IND-enabling studies, safety signals, accelerated-approval eligibility, prior precedents",
    },
    {
        "id": "investor",
        "name": "Biotech Investor",
        "lens": "TAM, time-to-clinic, capital intensity, syndicate appeal, comparable exits, IP defensibility",
    },
    {
        "id": "kol",
        "name": "Academic KOL",
        "lens": "publishability, where in the field this slots, who would champion or attack it, conference reception",
    },
    {
        "id": "patient",
        "name": "Patient Advocate",
        "lens": "lived experience, access, side-effect burden, real-world treatment journey, patient-reported outcomes",
    },
    {
        "id": "payer",
        "name": "Insurance Payer",
        "lens": "cost-effectiveness ratio, ICER thresholds, coverage policy, prior-auth complexity, budget impact",
    },
    {
        "id": "short_seller",
        "name": "Adversarial Short-Seller",
        "lens": "what's the fatal flaw, where does the science fall apart, what would make this collapse in trial readouts",
    },
]


def archetype_prompt(archetype: dict, paper_text: str) -> list[dict]:
    return [
        {"role": "system", "content": DOMAIN_FRAME},
        {
            "role": "user",
            "content": (
                f"You are a {archetype['name']}. Your evaluation lens: {archetype['lens']}.\n\n"
                "Read the paper below and reply with a JSON object:\n"
                "{\n"
                '  "scores": { "clinical_viability": 1-10, "regulatory_risk": 1-10,\n'
                '              "market_potential": 1-10, "patient_impact": 1-10,\n'
                '              "novelty": 1-10, "falsifiability": 1-10 },\n'
                '  "verdict": "<200 word verdict in your archetype voice>",\n'
                '  "would_move_price": "<one sentence on what catalyst would shift '
                'the prediction-market price for this paper, up or down>"\n'
                "}\n"
                "Score honestly from your lens — adversarial archetypes should be adversarial.\n\n"
                "PAPER:\n---\n" + paper_text[:10000] + "\n---"
            ),
        },
    ]
