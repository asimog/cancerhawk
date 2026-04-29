"""Writes a completed CancerHawk block to ``results/block-N/`` and
rewrites ``results/index.html`` so GitHub Pages always shows the latest.

Each block directory contains:
  - paper.md            (raw markdown of the paper)
  - paper.html          (standalone rendered paper page)
  - analysis.json       (full archetype analysis + market price + topics)
  - block.json          (run metadata: research_goal, models, timestamps)

``results/index.html`` lists all blocks chronologically and embeds the
latest block's paper + visualizations inline.
"""

from __future__ import annotations

import html
import json
import re
import subprocess
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
BLOCK_DIR_RE = re.compile(r"^block-(\d+)$")
PUBLIC_BASE_URL = "https://asimog.github.io/cancerhawk"


def next_block_number() -> int:
    if not RESULTS_DIR.is_dir():
        return 1
    nums = []
    for entry in RESULTS_DIR.iterdir():
        m = BLOCK_DIR_RE.match(entry.name)
        if m:
            nums.append(int(m.group(1)))
    return (max(nums) + 1) if nums else 1


def publish_block(
    paper,
    analysis,
    derived_topics: list[dict],
    research_goal: str,
    models: dict,
    peer_reviews: list[dict] | None = None,
    simulations: list[dict] | None = None,
) -> dict:
    """Write block-N/ + rewrite results/index.html. Returns metadata."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    block_n = next_block_number()
    block_dir = RESULTS_DIR / f"block-{block_n}"
    block_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).isoformat()

    # paper.md
    (block_dir / "paper.md").write_text(paper.full_text(), encoding="utf-8")

    # analysis.json
    analysis_payload = {
        "archetypes": analysis.archetypes,
        "market_price": analysis.market_price,
        "score_matrix": analysis.score_matrix,
        "consensus_dim": analysis.consensus_dim,
        "headline_catalysts": analysis.headline_catalysts,
        "derived_topics": derived_topics,
    }
    # Peer reviews and simulations (if provided)
    if peer_reviews is not None:
        analysis_payload["peer_reviews"] = peer_reviews
    if simulations is not None:
        analysis_payload["simulations"] = simulations
    (block_dir / "analysis.json").write_text(
        json.dumps(analysis_payload, indent=2), encoding="utf-8"
    )

    # block.json
    block_meta = {
        "block": block_n,
        "title": paper.title,
        "research_goal": research_goal,
        "models": models,
        "timestamp": timestamp,
        "section_count": len(paper.sections),
        "accepted_submissions": len(paper.accepted_submissions),
        "rejection_count": len(paper.rejections),
        "market_price": analysis.market_price,
        "has_peer_review": peer_reviews is not None and len(peer_reviews) > 0,
        "has_simulations": simulations is not None and len(simulations) > 0,
    }
    (block_dir / "block.json").write_text(json.dumps(block_meta, indent=2), encoding="utf-8")

    # Standalone paper.html for the block
    (block_dir / "paper.html").write_text(
        _render_block_page(paper, analysis_payload, block_meta), encoding="utf-8"
    )

    # Rewrite results/index.html showing latest + history
    _rewrite_index()

    return {"block": block_n, "path": str(block_dir.relative_to(REPO_ROOT)), "meta": block_meta}


def load_previous_block_context(limit: int = 3) -> str:
    """Return concise prior-block context for Block N+1 generation."""
    blocks = _load_blocks()
    if not blocks:
        return ""
    context_blocks = blocks[:limit]
    chunks = []
    for n, meta, analysis, paper_md in context_blocks:
        abstract = _paper_excerpt(paper_md, max_chars=1800)
        topics = analysis.get("derived_topics", [])[:3]
        topic_lines = "\n".join(
            f"  - {t.get('title', '')}: {t.get('rationale', '')}" for t in topics
        )
        chunks.append(
            f"[CancerHawk Block {n}] {meta.get('title', 'Untitled')}\n"
            f"URL: {PUBLIC_BASE_URL}/block-{n}/paper.html\n"
            f"Research goal: {meta.get('research_goal', '')}\n"
            f"Market confidence: {int(float(meta.get('market_price', 0)) * 100)}%\n"
            f"Useful prior findings:\n{abstract}\n"
            f"Next-topic seeds:\n{topic_lines or '  - None recorded.'}"
        )
    return "\n\n".join(chunks)


def _rewrite_index() -> None:
    blocks = _load_blocks()
    if blocks:
        latest = blocks[0]
        index_html = _render_index(latest, blocks)
    else:
        index_html = _empty_index()
    (RESULTS_DIR / "index.html").write_text(index_html, encoding="utf-8")
    (RESULTS_DIR / "blocks.html").write_text(_render_archive(blocks), encoding="utf-8")
    (RESULTS_DIR / "run.html").write_text(_render_run_page(), encoding="utf-8")


def _load_blocks() -> list[tuple[int, dict, dict, str]]:
    blocks = []
    if not RESULTS_DIR.is_dir():
        return blocks
    for entry in RESULTS_DIR.iterdir():
        m = BLOCK_DIR_RE.match(entry.name)
        if not m:
            continue
        meta_path = entry / "block.json"
        analysis_path = entry / "analysis.json"
        paper_path = entry / "paper.md"
        if not meta_path.is_file() or not analysis_path.is_file() or not paper_path.is_file():
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
        paper_md = paper_path.read_text(encoding="utf-8")
        blocks.append((int(m.group(1)), meta, analysis, paper_md))
    blocks.sort(key=lambda x: x[0], reverse=True)
    return blocks


# ===== HTML rendering =====


def _render_peer_reviews(peer_reviews: list[dict]) -> str:
    """Render the peer reviews section with tabbed interface."""
    if not peer_reviews:
        return '<p class="muted">No peer reviews available.</p>'

    # Compute acceptance probability from individual reviews
    accept_weight = {
        "accept": 1.0,
        "minor_revision": 0.7,
        "major_revision": 0.3,
        "reject": 0.0,
    }
    conf_sum = 0.0
    weighted_sum = 0.0
    for r in peer_reviews:
        rec = r.get("recommendation", "major_revision").lower()
        conf = _safe_float(r.get("confidence"), 0.7, lower=0.0, upper=1.0)
        weight = accept_weight.get(rec, 0.3)
        weighted_sum += weight * conf
        conf_sum += conf
    acceptance_probability = weighted_sum / conf_sum if conf_sum else 0.0

    # Banner
    banner = (
        f'<div class="acceptance-banner" style="margin-bottom:16px;padding:12px;'
        f'background:rgba(111,219,111,0.1);border:1px solid #1b5e20;border-radius:8px;">'
        f'<strong>Peer review acceptance probability:</strong> {acceptance_probability:.0%}'
        f'</div>'
    )

    reviews_html = []
    for idx, r in enumerate(peer_reviews):
        archetype_name = html.escape(r.get("archetype_name", "Unknown"))
        rec = r.get("recommendation", "major_revision").lower()
        # CSS class uses the first word (accept/minor/major/reject)
        rec_class = rec.split("_")[0]
        confidence = _safe_float(r.get("confidence"), 0.7, lower=0.0, upper=1.0)
        summary = html.escape(r.get("summary", ""))

        # Dimension scores
        dims = r.get("dimension_scores", {})
        dims_html = "".join(
            f'<div class="dim-score"><div class="dim-name">{html.escape(str(dim))}</div>'
            f'<div class="dim-value">{_score_text(score)}</div></div>'
            for dim, score in dims.items()
        )

        # Lists
        criticisms = "".join(f"<li>{html.escape(c)}</li>" for c in r.get("criticisms", []))
        fixes = "".join(f"<li>{html.escape(f)}</li>" for f in r.get("required_fixes", []))
        experiments = "".join(f"<li>{html.escape(e)}</li>" for e in r.get("suggested_experiments", []))

        reviews_html.append(
            f'<div class="peer-review-card" id="review-{idx}">'
            f'<div class="peer-review-header">'
            f'<span class="peer-review-archetype">{archetype_name}</span>'
            f'<span class="peer-review-rec {rec_class}">{rec.replace("_", " ")}</span>'
            f'</div>'
            f'<div class="peer-review-confidence">Confidence: {confidence:.0%}</div>'
            f'<p><strong>Summary:</strong> {summary}</p>'
            f'<h4>Dimension scores</h4><div class="dimension-scores">{dims_html}</div>'
            f'<h4>Criticisms</h4><ul class="criticisms-list">{criticisms}</ul>'
            f'<h4>Required fixes</h4><ul class="fixes-list">{fixes}</ul>'
            f'<h4>Suggested experiments</h4><ul class="experiments-list">{experiments}</ul>'
            f'</div>'
        )

    return banner + "".join(reviews_html)


def _render_simulations(simulations: list[dict]) -> str:
    """Render the simulations proposals section."""
    if not simulations:
        return '<p class="muted">No simulation proposals recommended.</p>'

    sims_html = []
    scene_payloads = []
    for idx, s in enumerate(simulations):
        sim_id = _slugify(str(s.get("id") or f"simulation-{idx + 1}"))
        sim_type = html.escape(str(s.get("type", "html5_canvas")))
        title = html.escape(str(s.get("title") or f"Simulation {idx + 1}"))
        desc = html.escape(str(s.get("description", "No description provided.")))
        rationale = html.escape(str(s.get("rationale", "")))
        metrics = s.get("expected_metrics", [])
        metrics_html = "".join(f"<li>{html.escape(str(m))}</li>" for m in metrics)
        scene_payloads.append(
            {
                "id": sim_id,
                "title": str(s.get("title") or f"Simulation {idx + 1}"),
                "scene": str(s.get("scene") or "trajectory_manifold"),
                "seed": int(_safe_float(s.get("seed"), idx + 1)),
                "parameters": s.get("parameters") or {},
            }
        )

        sims_html.append(
            f'<div class="simulation-card" data-simulation="{sim_id}">'
            f'<div class="simulation-copy">'
            f'<div class="simulation-type">{sim_type}</div>'
            f'<h3>{title}</h3>'
            f'<p>{desc}</p>'
            f'<h4>Why this matters</h4><p>{rationale}</p>'
            f'<h4>Readouts</h4><ul>{metrics_html}</ul>'
            f'</div>'
            f'<div class="simulation-stage">'
            f'<canvas id="sim-{sim_id}" aria-label="{title} interactive HTML5 canvas simulation"></canvas>'
            f'<div class="simulation-overlay"><span>Native HTML5 canvas</span><strong>{title}</strong></div>'
            f'</div>'
            f'</div>'
        )

    return (
        '<div class="simulation-intro">'
        '<p>Runnable browser-native simulations generated after peer review. '
        'Each scene uses only inline HTML5 Canvas 2D to turn the paper into a falsifiable visual model.</p>'
        '</div>'
        + "".join(sims_html)
        + _simulation_script(scene_payloads)
    )


def _render_block_page(paper, analysis_payload: dict, meta: dict) -> str:
    abstract = next((s for s in paper.sections if s["heading"].lower() == "abstract"), None)
    sections_html = "\n".join(
        f'<section><h2>{html.escape(s["heading"])}</h2>'
        f'<div class="prose">{_md_inline_to_html(s["content"])}</div></section>'
        for s in paper.sections
        if s["heading"].lower() != "abstract"
    )
    archetype_table = _archetype_table(analysis_payload["archetypes"])
    topics_table = _topics_table(analysis_payload.get("derived_topics", []))
    catalysts_html = _catalysts_html(analysis_payload.get("headline_catalysts", []))
    peer_reviews = analysis_payload.get("peer_reviews", [])
    simulations = analysis_payload.get("simulations", [])
    peer_reviews_html = _render_peer_reviews(peer_reviews) if peer_reviews else ""
    simulations_html = _render_simulations(simulations) if simulations else ""
    return _PAGE_SHELL.format(
        title=html.escape(paper.title),
        block=meta["block"],
        timestamp=html.escape(meta["timestamp"]),
        market_price=analysis_payload["market_price"],
        market_pct=int(analysis_payload["market_price"] * 100),
        research_goal=html.escape(meta["research_goal"]),
        sections_html=sections_html,
        archetype_table=archetype_table,
        topics_table=topics_table,
        catalysts_html=catalysts_html,
        peer_reviews_html=peer_reviews_html,
        simulations_html=simulations_html,
        abstract_html=_abstract_html(abstract["content"] if abstract else ""),
        analysis_json=html.escape(json.dumps(analysis_payload, indent=2)),
        consensus_json=html.escape(json.dumps(analysis_payload["consensus_dim"])),
        score_matrix_json=html.escape(json.dumps(analysis_payload["score_matrix"])),
        block_history_html="",
        nav_html=_nav_html(meta["block"], in_block_page=True),
    )


def _render_index(latest, all_blocks) -> str:
    block_n, meta, analysis_payload, paper_md = latest
    # Reconstruct paper sections from markdown for display
    sections_html = _md_to_sections_html(paper_md, skip_headings={"abstract"})
    archetype_table = _archetype_table(analysis_payload["archetypes"])
    topics_table = _topics_table(analysis_payload.get("derived_topics", []))
    catalysts_html = _catalysts_html(analysis_payload.get("headline_catalysts", []))
    peer_reviews = analysis_payload.get("peer_reviews", [])
    simulations = analysis_payload.get("simulations", [])
    peer_reviews_html = _render_peer_reviews(peer_reviews) if peer_reviews else ""
    simulations_html = _render_simulations(simulations) if simulations else ""
    history_html = _history_html(all_blocks)
    return _PAGE_SHELL.format(
        title=html.escape(meta["title"]),
        block=meta["block"],
        timestamp=html.escape(meta["timestamp"]),
        market_price=analysis_payload["market_price"],
        market_pct=int(analysis_payload["market_price"] * 100),
        research_goal=html.escape(meta["research_goal"]),
        sections_html=sections_html,
        archetype_table=archetype_table,
        topics_table=topics_table,
        catalysts_html=catalysts_html,
        peer_reviews_html=peer_reviews_html,
        simulations_html=simulations_html,
        abstract_html=_abstract_html(_md_section_content(paper_md, "Abstract")),
        analysis_json=html.escape(json.dumps(analysis_payload, indent=2)),
        consensus_json=html.escape(json.dumps(analysis_payload["consensus_dim"])),
        score_matrix_json=html.escape(json.dumps(analysis_payload["score_matrix"])),
        block_history_html=history_html,
        nav_html=_nav_html(meta["block"], in_block_page=False),
    )


def _empty_index() -> str:
    return """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>CancerHawk — Research Evolution Record</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>body{font:16px/1.55 system-ui;max-width:880px;margin:0 auto;padding:24px;color:#0f0;background:#000}a{color:#0f7}</style>
</head><body>
<h1>CancerHawk · Research Evolution Record</h1>
<p>No blocks published yet. Start the local engine: <code>python app/main.py</code>, open <a href="http://localhost:8765">localhost:8765</a>, paste your OpenRouter key and run.</p>
</body></html>
"""


def _render_archive(blocks) -> str:
    cards = []
    for n, meta, analysis, paper_md in blocks:
        excerpt = html.escape(_paper_excerpt(paper_md, max_chars=420))
        title = html.escape(str(meta.get("title", "Untitled")))
        goal = html.escape(str(meta.get("research_goal", "")))
        timestamp = html.escape(str(meta.get("timestamp", ""))[:19])
        market_pct = int(float(analysis.get("market_price", meta.get("market_price", 0)) or 0) * 100)
        cards.append(
            f'<article class="archive-card">'
            f'<div class="archive-kicker">Block {n} · {timestamp} · {market_pct}%</div>'
            f'<h2><a href="block-{n}/paper.html">{title}</a></h2>'
            f'<p><strong>Research goal:</strong> {goal}</p>'
            f'<p>{excerpt}</p>'
            f'<p class="archive-links"><a href="block-{n}/paper.html">Open paper</a> '
            f'<a href="block-{n}/analysis.json">Analysis JSON</a> '
            f'<a href="block-{n}/paper.md">Markdown</a></p>'
            f'</article>'
        )
    body = "".join(cards) if cards else '<p class="muted">No blocks published yet.</p>'
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>CancerHawk Blocks</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
body{{font:16px/1.6 -apple-system, Segoe UI, Helvetica, Arial, sans-serif;max-width:1040px;margin:0 auto;padding:24px;background:#050a06;color:#c8e6c9}}
a{{color:#6fdb6f}} h1,h2{{color:#6fdb6f}} header{{border-bottom:1px solid #1a3a1a;margin-bottom:22px;padding-bottom:16px}}
.archive-card{{background:#0a1f0a;border:1px solid #1a3a1a;border-radius:16px;padding:18px;margin:16px 0}}
.archive-kicker{{color:#8fbf8f;font-size:13px;text-transform:uppercase;letter-spacing:.06em}}
.archive-links{{display:flex;gap:14px;flex-wrap:wrap}} .muted{{color:#6a8a6a}}
</style></head><body>
<header>
  <h1>CancerHawk Block Archive</h1>
  <p>All published research blocks. Future blocks can cite and extend these papers when appropriate.</p>
  <p><a href="./">Latest block</a> · <a href="run.html">Run a block</a></p>
</header>
{body}
</body></html>
"""


def _nav_html(block: int, *, in_block_page: bool) -> str:
    latest_href = "../" if in_block_page else "./"
    archive_href = "../blocks.html" if in_block_page else "blocks.html"
    run_href = "../run.html" if in_block_page else "run.html"
    permanent_href = "paper.html" if in_block_page else f"block-{block}/paper.html"
    return (
        '<nav class="site-nav">'
        f'<a href="{latest_href}">Latest block</a><a href="{archive_href}">All blocks</a>'
        f'<a href="{run_href}">Run a block</a>'
        f'<a href="{permanent_href}">Permanent link</a>'
        '</nav>'
    )


def _archetype_table(archetypes: list[dict]) -> str:
    if not archetypes:
        return '<p class="muted">No archetype results.</p>'
    rows = []
    for a in archetypes:
        scores = a.get("scores", {})
        verdict = html.escape((a.get("verdict") or "")[:600])
        rows.append(
            f'<tr><td><strong>{html.escape(a.get("archetype_name", ""))}</strong></td>'
            f'<td>{_score_text(scores.get("clinical_viability"))}</td>'
            f'<td>{_score_text(scores.get("regulatory_risk"))}</td>'
            f'<td>{_score_text(scores.get("market_potential"))}</td>'
            f'<td>{_score_text(scores.get("patient_impact"))}</td>'
            f'<td>{_score_text(scores.get("novelty"))}</td>'
            f'<td>{_score_text(scores.get("falsifiability"))}</td>'
            f'<td class="verdict">{verdict}</td></tr>'
        )
    return (
        '<table class="archetype"><thead><tr>'
        "<th>Archetype</th><th>Clin.Viab</th><th>Reg.Risk</th><th>Market</th>"
        "<th>Patient</th><th>Novelty</th><th>Falsif.</th><th>Verdict</th>"
        f"</tr></thead><tbody>{''.join(rows)}</tbody></table>"
    )


def _topics_table(topics: list[dict]) -> str:
    if not topics:
        return '<p class="muted">No derived topics.</p>'
    rows = []
    for t in topics:
        rows.append(
            f'<tr><td>{html.escape(str(t.get("id", "—")))}</td>'
            f'<td>{html.escape(str(t.get("title", "")))}</td>'
            f'<td>{html.escape(str(t.get("probability", "—")))}</td>'
            f'<td>{html.escape(str(t.get("impact", "—")))}</td>'
            f'<td>{html.escape(str(t.get("token_cost", "—")))}</td>'
            f'<td class="rationale">{html.escape(str(t.get("rationale", "")))}</td></tr>'
        )
    return (
        '<table class="topics"><thead><tr><th>#</th><th>Title</th><th>Prob</th>'
        "<th>Impact</th><th>Tokens</th><th>Rationale</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def _catalysts_html(catalysts: list[str]) -> str:
    if not catalysts:
        return ""
    items = "\n".join(f"<li>{html.escape(c)}</li>" for c in catalysts)
    return f'<ul class="catalysts">{items}</ul>'


def _simulation_script(scene_payloads: list[dict]) -> str:
    payload_json = _script_json(scene_payloads)
    template = """
<script type="application/json" id="simulation-scenes">__SCENE_PAYLOAD__</script>
<script>
(function () {
  const scenes = JSON.parse(document.getElementById('simulation-scenes')?.textContent || '[]');
  const palette = ['#6fdb6f', '#42c6ff', '#ffc857', '#ff6b6b', '#b892ff'];

  function seededRandom(seed) {
    let value = Math.max(1, seed || 1) % 2147483647;
    return function () {
      value = value * 16807 % 2147483647;
      return (value - 1) / 2147483646;
    };
  }

  function fitCanvas(canvas) {
    const box = canvas.parentElement.getBoundingClientRect();
    const ratio = Math.min(window.devicePixelRatio || 1, 2);
    const width = Math.max(320, Math.floor(box.width));
    const height = Math.max(280, Math.floor(box.height));
    if (canvas.width !== Math.floor(width * ratio) || canvas.height !== Math.floor(height * ratio)) {
      canvas.width = Math.floor(width * ratio);
      canvas.height = Math.floor(height * ratio);
    }
    const ctx = canvas.getContext('2d');
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    return { ctx, width, height };
  }

  function clear(ctx, width, height) {
    const gradient = ctx.createRadialGradient(width * 0.5, height * 0.42, 8, width * 0.5, height * 0.42, width * 0.72);
    gradient.addColorStop(0, 'rgba(111,219,111,0.16)');
    gradient.addColorStop(0.5, 'rgba(8,36,20,0.96)');
    gradient.addColorStop(1, 'rgba(5,10,6,1)');
    ctx.fillStyle = gradient;
    ctx.fillRect(0, 0, width, height);
    ctx.strokeStyle = 'rgba(111,219,111,0.08)';
    ctx.lineWidth = 1;
    for (let x = 0; x < width; x += 34) {
      ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, height); ctx.stroke();
    }
    for (let y = 0; y < height; y += 34) {
      ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(width, y); ctx.stroke();
    }
  }

  function dot(ctx, x, y, r, color, glow) {
    ctx.save();
    ctx.shadowBlur = glow || 16;
    ctx.shadowColor = color;
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(x, y, r, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();
  }

  function drawTrajectory(ctx, width, height, rand, time) {
    for (let strand = 0; strand < 5; strand++) {
      const color = palette[strand % palette.length];
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.shadowBlur = 12;
      ctx.shadowColor = color;
      ctx.beginPath();
      for (let i = 0; i < 86; i++) {
        const t = i / 10 + time * 0.22;
        const radius = 42 + strand * 17;
        const x = width * 0.5 + Math.sin(t + strand) * radius + Math.cos(t * 0.37) * 36;
        const y = height * 0.5 + Math.cos(t * 0.8 + strand) * (38 + strand * 9) + (strand - 2) * 18;
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
        if (i % 17 === 0) dot(ctx, x, y, 4 + rand() * 3, color, 18);
      }
      ctx.stroke();
      ctx.shadowBlur = 0;
    }
  }

  function drawCounterfactual(ctx, width, height, rand, time) {
    const mid = width * 0.5;
    ctx.strokeStyle = 'rgba(255,200,87,0.72)';
    ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(mid, 32); ctx.lineTo(mid, height - 32); ctx.stroke();
    ctx.fillStyle = 'rgba(255,200,87,0.12)';
    ctx.fillRect(mid - 28 - Math.sin(time * 2) * 8, 28, 56 + Math.sin(time * 2) * 16, height - 56);
    for (let i = 0; i < 42; i++) {
      const a = i * 0.48 + time * 0.55;
      dot(ctx, mid - 105 + Math.cos(a) * 62, height * 0.5 + Math.sin(a * 1.3) * 76, 5, '#42c6ff', 14);
      const spread = Math.max(12, i * 2.2 + Math.sin(time * 1.4) * 10);
      const color = i > 18 ? '#ff6b6b' : '#6fdb6f';
      dot(ctx, mid + 92 + Math.cos(a) * spread, height * 0.5 + Math.sin(a * 1.45) * spread * 0.8, 5, color, 15);
    }
    ctx.fillStyle = '#c8e6c9';
    ctx.font = '12px ui-monospace, monospace';
    ctx.fillText('control', mid - 145, 28);
    ctx.fillText('treated perturbation', mid + 42, 28);
  }

  function drawGradient(ctx, width, height, rand, time) {
    const gradient = ctx.createLinearGradient(0, 0, width, height);
    gradient.addColorStop(0, 'rgba(66,198,255,0.16)');
    gradient.addColorStop(0.5, 'rgba(111,219,111,0.12)');
    gradient.addColorStop(1, 'rgba(255,107,107,0.2)');
    ctx.fillStyle = gradient;
    ctx.fillRect(0, 0, width, height);
    for (let i = 0; i < 90; i++) {
      const base = i * 997 + (scenes.length || 1);
      const x = ((base * 37) % width) + Math.sin(time + i) * 16;
      const y = ((base * 61) % height) + Math.cos(time * 0.8 + i) * 12;
      const stress = (x + y) / (width + height);
      const color = stress > 0.62 ? '#ff6b6b' : stress > 0.38 ? '#ffc857' : '#6fdb6f';
      dot(ctx, x, y, 3 + stress * 5, color, 12);
    }
    ctx.strokeStyle = 'rgba(200,230,201,0.24)';
    ctx.lineWidth = 1;
    for (let i = 0; i < 7; i++) {
      ctx.beginPath();
      const y = height * (i + 1) / 8 + Math.sin(time + i) * 8;
      ctx.moveTo(18, y);
      ctx.bezierCurveTo(width * 0.25, y - 42, width * 0.72, y + 42, width - 18, y);
      ctx.stroke();
    }
  }

  function boot(spec) {
    const canvas = document.getElementById('sim-' + spec.id);
    if (!canvas) return;
    const rand = seededRandom(spec.seed);
    function frame(ms) {
      if (!canvas.offsetParent) {
        requestAnimationFrame(frame);
        return;
      }
      const time = ms / 1000;
      const fitted = fitCanvas(canvas);
      clear(fitted.ctx, fitted.width, fitted.height);
      if (spec.scene === 'counterfactual_perturbation') drawCounterfactual(fitted.ctx, fitted.width, fitted.height, rand, time);
      else if (spec.scene === 'microenvironment_gradient') drawGradient(fitted.ctx, fitted.width, fitted.height, rand, time);
      else drawTrajectory(fitted.ctx, fitted.width, fitted.height, rand, time);
      requestAnimationFrame(frame);
    }
    requestAnimationFrame(frame);
  }

  scenes.forEach(boot);
})();
</script>
"""
    return template.replace("__SCENE_PAYLOAD__", payload_json)

def _script_json(value) -> str:
    return (
        json.dumps(value)
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
    )


def _safe_float(value, default: float, *, lower: float | None = None, upper: float | None = None) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = default
    if lower is not None:
        number = max(lower, number)
    if upper is not None:
        number = min(upper, number)
    return number


def _score_text(value) -> str:
    if value in (None, ""):
        return "—"
    try:
        return str(int(float(value)))
    except (TypeError, ValueError):
        return html.escape(str(value))


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "simulation"


def _history_html(blocks) -> str:
    if len(blocks) <= 1:
        return ""
    rows = []
    for n, meta, analysis, _ in blocks:
        rows.append(
            f'<tr><td><a href="block-{n}/paper.html">Block {n}</a></td>'
            f'<td>{html.escape(meta["title"])[:80]}</td>'
            f'<td>{int(analysis["market_price"] * 100)}%</td>'
            f'<td>{html.escape(meta["timestamp"][:19])}</td></tr>'
        )
    return (
        '<section><h2>All blocks</h2><table class="history"><thead><tr>'
        "<th>Block</th><th>Title</th><th>Mkt</th><th>Timestamp</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></section>"
    )


def _md_span_to_html(text: str) -> str:
    escaped = html.escape(text)
    escaped = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", escaped)
    escaped = re.sub(r"`([^`]+)`", r"<code>\1</code>", escaped)
    return escaped


def _split_table_row(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def _is_table_separator(line: str) -> bool:
    cells = _split_table_row(line)
    return bool(cells) and all(re.fullmatch(r":?-{3,}:?", cell or "") for cell in cells)


def _md_inline_to_html(text: str) -> str:
    lines = text.splitlines()
    html_blocks: list[str] = []
    paragraph: list[str] = []

    def flush_paragraph() -> None:
        if not paragraph:
            return
        html_blocks.append(f"<p>{_md_span_to_html(' '.join(line.strip() for line in paragraph))}</p>")
        paragraph.clear()

    i = 0
    while i < len(lines):
        line = lines[i].rstrip()
        stripped = line.strip()
        if not stripped:
            flush_paragraph()
            i += 1
            continue

        if (
            stripped.startswith("|")
            and i + 1 < len(lines)
            and lines[i + 1].strip().startswith("|")
            and _is_table_separator(lines[i + 1].strip())
        ):
            flush_paragraph()
            headers = _split_table_row(stripped)
            i += 2
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                rows.append(_split_table_row(lines[i].strip()))
                i += 1
            thead = "".join(f"<th>{_md_span_to_html(cell)}</th>" for cell in headers)
            body_rows = []
            for row in rows:
                cells = row + [""] * max(0, len(headers) - len(row))
                body_rows.append(
                    "<tr>" + "".join(f"<td>{_md_span_to_html(cell)}</td>" for cell in cells[: len(headers)]) + "</tr>"
                )
            html_blocks.append(
                '<table class="paper-table"><thead><tr>'
                + thead
                + "</tr></thead><tbody>"
                + "".join(body_rows)
                + "</tbody></table>"
            )
            continue

        if re.match(r"^\d+\.\s+", stripped):
            flush_paragraph()
            items = []
            while i < len(lines) and re.match(r"^\d+\.\s+", lines[i].strip()):
                items.append(re.sub(r"^\d+\.\s+", "", lines[i].strip()))
                i += 1
            html_blocks.append("<ol>" + "".join(f"<li>{_md_span_to_html(item)}</li>" for item in items) + "</ol>")
            continue

        if stripped.startswith("- "):
            flush_paragraph()
            items = []
            while i < len(lines) and lines[i].strip().startswith("- "):
                items.append(lines[i].strip()[2:])
                i += 1
            html_blocks.append("<ul>" + "".join(f"<li>{_md_span_to_html(item)}</li>" for item in items) + "</ul>")
            continue

        paragraph.append(stripped)
        i += 1

    flush_paragraph()
    return "\n".join(html_blocks)


def _paper_excerpt(md: str, max_chars: int = 1200) -> str:
    lines = []
    for line in md.splitlines():
        if line.startswith("#"):
            continue
        clean = line.strip()
        if clean:
            lines.append(clean)
        if len(" ".join(lines)) >= max_chars:
            break
    excerpt = " ".join(lines)
    return (excerpt[: max_chars - 1].rstrip() + "…") if len(excerpt) > max_chars else excerpt


def _md_section_content(md: str, heading: str) -> str:
    target = heading.strip().lower()
    current = None
    buf: list[str] = []
    for line in md.splitlines():
        if line.startswith("## "):
            if current == target:
                break
            current = line[3:].strip().lower()
            buf = []
            continue
        if current == target:
            buf.append(line)
    return "\n".join(buf).strip()


def _abstract_html(content: str) -> str:
    if not content.strip():
        return ""
    return (
        '<section class="abstract-summary"><h2>Abstract</h2>'
        f'<div class="prose">{_md_inline_to_html(content)}</div></section>'
    )


def _md_to_sections_html(md: str, skip_headings: set[str] | None = None) -> str:
    skip = {h.lower() for h in (skip_headings or set())}
    lines = md.splitlines()
    sections = []
    cur_heading = None
    cur_buf = []
    for line in lines:
        if line.startswith("## "):
            if cur_heading is not None:
                sections.append((cur_heading, "\n".join(cur_buf).strip()))
            cur_heading = line[3:].strip()
            cur_buf = []
        elif line.startswith("# "):
            continue  # title handled separately
        else:
            cur_buf.append(line)
    if cur_heading is not None:
        sections.append((cur_heading, "\n".join(cur_buf).strip()))
    return "\n".join(
        f'<section><h2>{html.escape(h)}</h2>'
        f'<div class="prose">{_md_inline_to_html(c)}</div></section>'
        for h, c in sections
        if h.lower() not in skip
    )


def try_git_publish(block_n: int) -> str:
    msg = f"publish: block {block_n}"
    try:
        subprocess.run(["git", "add", "-f", "results"], cwd=REPO_ROOT, check=True, capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", msg], cwd=REPO_ROOT, check=True, capture_output=True
        )
        subprocess.run(["git", "push"], cwd=REPO_ROOT, check=True, capture_output=True)
        return f"pushed block {block_n}"
    except subprocess.CalledProcessError as exc:
        return f"git failed: {exc.stderr.decode(errors='replace')[:200] if exc.stderr else exc}"
    except FileNotFoundError:
        return "git not available"


def _render_run_page() -> str:
    return """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>Run a CancerHawk Block</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
:root{color-scheme:dark}
*{box-sizing:border-box}
body{font:16px/1.6 -apple-system,Segoe UI,Helvetica,Arial,sans-serif;margin:0;background:#050a06;color:#c8e6c9}
header{padding:24px clamp(18px,4vw,44px);border-bottom:1px solid #1a3a1a;background:#071407}
h1{color:#6fdb6f;margin:0 0 8px;line-height:1.15}
a{color:#6fdb6f}
.site-nav{display:flex;flex-wrap:wrap;gap:10px;margin-top:14px}
.site-nav a{color:#071407;background:#6fdb6f;border-radius:999px;padding:6px 11px;text-decoration:none;font-size:13px;font-weight:700}
main{padding:24px clamp(18px,4vw,44px)}
.status{border:1px solid #1a3a1a;background:#0a1f0a;border-radius:10px;padding:14px 16px;margin-bottom:18px}
.status strong{color:#d7ffd7}
.actions{display:flex;flex-wrap:wrap;gap:12px;margin:14px 0}
.button{border:1px solid #6fdb6f;background:#6fdb6f;color:#061006;border-radius:8px;padding:10px 13px;text-decoration:none;font-weight:700}
.button.secondary{background:#071407;color:#c8e6c9;border-color:#1a3a1a}
code,pre{font-family:ui-monospace,Menlo,Consolas,monospace}
pre{background:#071407;border:1px solid #1a3a1a;border-radius:10px;padding:14px;overflow:auto}
.backend-frame{width:100%;height:78vh;border:1px solid #1a3a1a;border-radius:12px;background:#000;display:none}
.notes{max-width:920px;color:#a4c4a4}
</style></head><body>
<header>
  <h1>Run a CancerHawk Block</h1>
  <p>Generate the next research block from your own OpenRouter API key using the local backend.</p>
  <nav class="site-nav"><a href="./">Latest block</a><a href="blocks.html">All blocks</a></nav>
</header>
<main>
  <section class="status" id="status"><strong>Checking local backend...</strong></section>
  <div class="actions">
    <a class="button" href="http://localhost:8765" target="_blank" rel="noreferrer">Open local backend</a>
    <button class="button secondary" type="button" id="retry">Check again</button>
  </div>
  <iframe class="backend-frame" id="backend" title="CancerHawk local backend" src="about:blank"></iframe>
  <section class="notes">
    <h2>Start the backend</h2>
    <p>GitHub Pages is static, so it cannot run Python by itself. Start CancerHawk locally, then this page will display the backend here.</p>
    <pre>uv run --with-requirements app/requirements.txt --with-requirements app/requirements-dev.txt python -m app.main</pre>
    <p>Once the backend is running, paste your OpenRouter API key, choose models, and run a fresh block. Completed blocks publish into <code>results/block-N/</code>. If you enable git publishing locally, the backend can commit and push the generated block to GitHub Pages.</p>
  </section>
</main>
<script>
const statusEl = document.getElementById('status');
const frame = document.getElementById('backend');
async function checkBackend(){
  statusEl.innerHTML = '<strong>Checking local backend...</strong>';
  try {
    const res = await fetch('http://localhost:8765/api/health', {cache:'no-store'});
    if (!res.ok) throw new Error('health check failed');
    statusEl.innerHTML = '<strong>Backend detected.</strong> The local CancerHawk control panel is embedded below.';
    frame.src = 'http://localhost:8765';
    frame.style.display = 'block';
  } catch (err) {
    statusEl.innerHTML = '<strong>Backend not running.</strong> Start it locally, then check again.';
    frame.style.display = 'none';
  }
}
document.getElementById('retry').addEventListener('click', checkBackend);
checkBackend();
</script>
</body></html>
"""


_PAGE_SHELL = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>{title} — CancerHawk Block {block}</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
:root {{ color-scheme: dark; }}
* {{ box-sizing: border-box; }}
body {{ font: 16px/1.6 -apple-system, Segoe UI, Helvetica, Arial, sans-serif; max-width: 1100px; margin: 0 auto; padding: 24px; background: #050a06; color: #c8e6c9; }}
header {{ border-bottom: 1px solid #1a3a1a; padding-bottom: 16px; margin-bottom: 24px; }}
.site-nav {{ display: flex; flex-wrap: wrap; gap: 10px; margin: 14px 0 0; }}
.site-nav a {{ color: #071407; background: #6fdb6f; border-radius: 999px; padding: 6px 11px; text-decoration: none; font-size: 13px; font-weight: 700; }}
h1 {{ color: #6fdb6f; margin: 0 0 8px 0; line-height: 1.2; }}
h2 {{ color: #6fdb6f; border-bottom: 1px solid #1a3a1a; padding-bottom: 6px; margin-top: 32px; }}
h3 {{ color: #6fdb6f; font-size: 15px; margin-top: 20px; }}
.meta {{ color: #6a8a6a; font-size: 14px; }}
.market-banner {{ background: #0a1f0a; border: 1px solid #1a3a1a; border-radius: 8px; padding: 16px 20px; margin: 16px 0; display: flex; align-items: center; gap: 24px; flex-wrap: wrap; }}
.market-banner .price {{ font-size: 36px; font-weight: 700; color: #6fdb6f; }}
.market-banner .label {{ font-size: 12px; color: #6a8a6a; text-transform: uppercase; letter-spacing: 1px; }}
.disclaimer {{ background: #1a0a0a; border-left: 3px solid #c44; padding: 10px 14px; font-size: 13px; margin: 16px 0; color: #f8b8b8; }}
section {{ margin: 28px 0; }}
.prose p {{ margin: 0 0 14px 0; }}
.page-tabs {{ display: flex; flex-wrap: wrap; gap: 8px; margin: 18px 0 22px; border-bottom: 1px solid #1a3a1a; padding-bottom: 10px; }}
.page-tab {{ background: #071407; border: 1px solid #1a3a1a; color: #a4c4a4; padding: 8px 14px; border-radius: 999px; cursor: pointer; }}
.page-tab.active {{ background: #6fdb6f; border-color: #6fdb6f; color: #061006; font-weight: 700; }}
.page-content {{ display: none; }}
.page-content.active {{ display: block; }}
.charts {{ display: grid; grid-template-columns: 1fr 1fr; gap: 18px; margin: 16px 0; }}
.chart-box {{ background: #0a1f0a; border: 1px solid #1a3a1a; border-radius: 8px; padding: 16px; }}
.chart-box h3 {{ margin: 0 0 12px 0; color: #6fdb6f; font-size: 14px; text-transform: uppercase; letter-spacing: 1px; }}
table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
th, td {{ padding: 8px 10px; border-bottom: 1px solid #1a3a1a; text-align: left; vertical-align: top; }}
th {{ color: #6fdb6f; font-weight: 600; font-size: 12px; text-transform: uppercase; letter-spacing: 0.5px; }}
td.verdict, td.rationale {{ color: #c8e6c9; max-width: 320px; font-size: 13px; line-height: 1.4; }}
.paper-table {{ margin: 16px 0 20px; background: #071407; border: 1px solid #1a3a1a; border-radius: 8px; overflow: hidden; display: block; overflow-x: auto; }}
.paper-table th, .paper-table td {{ min-width: 150px; }}
.abstract-summary {{ background: #071407; border: 1px solid #1a3a1a; border-radius: 12px; padding: 18px; margin: 18px 0 22px; }}
.abstract-summary h2 {{ margin-top: 0; }}
.catalysts {{ background: #0a1f0a; border-radius: 8px; padding: 12px 28px; }}
.catalysts li {{ margin: 6px 0; color: #c8e6c9; }}
.muted {{ color: #6a8a6a; }}
details {{ margin: 16px 0; }}
details summary {{ cursor: pointer; color: #6fdb6f; }}
pre {{ white-space: pre-wrap; word-wrap: break-word; background: #0a1f0a; padding: 12px; border-radius: 6px; font: 12px/1.5 ui-monospace, Menlo, monospace; }}
@media (max-width: 720px) {{ .charts {{ grid-template-columns: 1fr; }} }}

/* Peer Review styles */
.peer-reviews-tabs {{ display: flex; gap: 8px; margin-bottom: 18px; border-bottom: 1px solid #1a3a1a; padding-bottom: 8px; }}
.peer-reviews-tab {{ background: none; border: none; color: #a4c4a4; padding: 8px 16px; cursor: pointer; font-size: 14px; border-radius: 6px 6px 0 0; }}
.peer-reviews-tab.active {{ background: #0a1f0a; color: #6fdb6f; font-weight: 600; }}
.peer-review-card {{ background: #0a1f0a; border: 1px solid #1a3a1a; border-radius: 8px; padding: 14px; margin-bottom: 12px; }}
.peer-review-header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; }}
.peer-review-archetype {{ font-weight: 700; color: #c8e6c9; font-size: 15px; }}
.peer-review-rec {{ padding: 4px 10px; border-radius: 4px; font-size: 12px; font-weight: 600; text-transform: uppercase; }}
.peer-review-rec.accept {{ background: #1b5e20; color: #a5d6a7; }}
.peer-review-rec.minor {{ background: #f57f17; color: #fff; }}
.peer-review-rec.major {{ background: #c62828; color: #ff8a80; }}
.peer-review-rec.reject {{ background: #b71c1c; color: #ef9a9a; }}
.peer-review-confidence {{ font-size: 12px; color: #6a8a6a; }}
.dimension-scores {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 8px; margin: 12px 0; }}
.dim-score {{ background: #050a06; padding: 8px; border-radius: 4px; border: 1px solid #1a3a1a; text-align: center; }}
.dim-score .dim-name {{ font-size: 11px; color: #a4c4a4; text-transform: uppercase; }}
.dim-score .dim-value {{ font-size: 18px; font-weight: 700; color: #6fdb6f; }}
.criticisms-list, .fixes-list, .experiments-list {{ margin: 10px 0; padding-left: 20px; }}
.criticisms-list li, .fixes-list li, .experiments-list li {{ margin-bottom: 6px; color: #c8e6c9; }}
.simulation-intro {{ background: linear-gradient(135deg, rgba(111,219,111,0.12), rgba(66,198,255,0.08)); border: 1px solid #1a3a1a; border-radius: 14px; padding: 14px 16px; margin-bottom: 18px; }}
.simulation-card {{ display: grid; grid-template-columns: minmax(240px, 0.82fr) minmax(320px, 1.18fr); gap: 18px; background: #0a1f0a; border: 1px solid #1a3a1a; border-radius: 16px; padding: 16px; margin-bottom: 18px; border-left: 3px solid #42c6ff; overflow: hidden; }}
.simulation-type {{ font-size: 11px; color: #42c6ff; text-transform: uppercase; letter-spacing: 0.5px; font-weight: 600; }}
.simulation-copy h3 {{ margin: 6px 0 10px; font-size: 20px; color: #d7ffd7; }}
.simulation-copy h4 {{ margin-bottom: 4px; }}
.simulation-stage {{ position: relative; height: 320px; border-radius: 14px; overflow: hidden; background: radial-gradient(circle at 50% 40%, rgba(111,219,111,0.18), rgba(5,10,6,0.96) 62%); border: 1px solid rgba(111,219,111,0.24); overflow-anchor: none; }}
.simulation-stage canvas {{ width: 100%; height: 100%; display: block; }}
.simulation-overlay {{ position: absolute; left: 14px; right: 14px; bottom: 12px; display: flex; justify-content: space-between; gap: 10px; align-items: center; color: #d7ffd7; font-size: 12px; text-shadow: 0 1px 8px #000; pointer-events: none; }}
.simulation-overlay span {{ color: #42c6ff; text-transform: uppercase; letter-spacing: 0.08em; }}
.simulation-overlay strong {{ text-align: right; max-width: 60%; }}
@media (max-width: 860px) {{ .simulation-card {{ grid-template-columns: 1fr; }} .simulation-stage {{ height: 280px; }} }}
</style>
</head><body>
<header>
  <h1>{title}</h1>
  <p class="meta">CancerHawk · Block {block} · {timestamp}</p>
  <p class="meta"><strong>Research goal:</strong> {research_goal}</p>
  {nav_html}
</header>

<aside class="disclaimer">
  Autonomously generated by CancerHawk. This paper has undergone automated peer review by MiroShark archetype agents. May contain incorrect, incomplete, or fabricated claims. Independently verify before acting on any content.
</aside>

{abstract_html}

<!-- Page Tabs -->
<div class="page-tabs">
  <button class="page-tab active" data-tab="paper">Paper</button>
  <button class="page-tab" data-tab="peer-reviews">Peer Reviews</button>
  <button class="page-tab" data-tab="simulations">Simulations</button>
  <button class="page-tab" data-tab="analysis">Market Analysis</button>
</div>

<!-- Paper Content -->
<div id="paper-tab" class="page-content active">
{sections_html}
</div>

<!-- Peer Reviews Tab -->
<div id="peer-reviews-tab" class="page-content">
{peer_reviews_html}
</div>

<!-- Simulations Tab -->
<div id="simulations-tab" class="page-content">
{simulations_html}
</div>

<!-- Market Analysis Tab -->
<div id="analysis-tab" class="page-content">
<div class="market-banner">
  <div><div class="label">Synthesis market price</div><div class="price">{market_pct}%</div></div>
  <div><div class="label">Verdict</div><div>Aggregated archetype confidence in clinical + commercial viability</div></div>
</div>

<section>
  <h2>Visualizations</h2>
  <div class="charts">
    <div class="chart-box"><h3>Archetype score radar</h3><canvas id="radar"></canvas></div>
    <div class="chart-box"><h3>Consensus dimension scores</h3><canvas id="bars"></canvas></div>
    <div class="chart-box"><h3>Per-archetype average</h3><canvas id="archAvg"></canvas></div>
    <div class="chart-box"><h3>Synthesis-market price</h3><canvas id="price"></canvas></div>
  </div>
</section>

<section>
  <h2>What would move the price</h2>
  {catalysts_html}
</section>

<section>
  <h2>Archetype panel</h2>
  {archetype_table}
</section>

<section>
  <h2>Next-block topics derived</h2>
  {topics_table}
</section>
</div>

{block_history_html}

<details><summary>Full analysis JSON</summary><pre>{analysis_json}</pre></details>

<script>
const consensus = JSON.parse(document.querySelector('script[data-consensus]')?.textContent || "{consensus_json}".replace(/&quot;/g,'"'));
const matrix = JSON.parse("{score_matrix_json}".replace(/&quot;/g,'"'));
const marketPrice = {market_price};

const dimLabels = Object.keys(consensus);
const dimVals = Object.values(consensus);

// Chart.js charts (existing radar, bars, archAvg, price)
new Chart(document.getElementById('radar'), {{
  type: 'radar',
  data: {{
    labels: Object.keys(matrix),
    datasets: dimLabels.map((dim, i) => ({{
      label: dim,
      data: Object.values(matrix).map(s => s[dim] ?? 0),
      backgroundColor: `hsla({{(i*55)%360}}, 70%, 50%, 0.15)`,
      borderColor: `hsla({{(i*55)%360}}, 70%, 60%, 0.9)`,
      borderWidth: 1.5
    }}))
  }},
  options: {{ scales: {{ r: {{ beginAtZero: true, max: 10, grid: {{ color: '#1a3a1a' }}, angleLines: {{ color: '#1a3a1a' }}, pointLabels: {{ color: '#c8e6c9' }} }} }}, plugins: {{ legend: {{ labels: {{ color: '#c8e6c9', font: {{ size: 10 }} }} }} }} }}
}});

new Chart(document.getElementById('bars'), {{
  type: 'bar',
  data: {{ labels: dimLabels, datasets: [{{ label: 'Mean score', data: dimVals, backgroundColor: '#6fdb6f88', borderColor: '#6fdb6f', borderWidth: 1 }}] }},
  options: {{ scales: {{ y: {{ beginAtZero: true, max: 10, ticks: {{ color: '#c8e6c9' }} }}, x: {{ ticks: {{ color: '#c8e6c9' }} }} }}, plugins: {{ legend: {{ display: false }} }} }}
}});

const archIds = Object.keys(matrix);
const archAvgs = archIds.map(id => {{
  const v = Object.values(matrix[id] || {{}}).filter(x => typeof x === 'number');
  return v.length ? v.reduce((a,b)=>a+b,0)/v.length : 0;
}});
new Chart(document.getElementById('archAvg'), {{
  type: 'bar',
  data: {{ labels: archIds, datasets: [{{ label: 'Avg score', data: archAvgs, backgroundColor: '#3a8f3a99', borderColor: '#6fdb6f', borderWidth: 1 }}] }},
  options: {{ indexAxis: 'y', scales: {{ x: {{ beginAtZero: true, max: 10, ticks: {{ color: '#c8e6c9' }} }}, y: {{ ticks: {{ color: '#c8e6c9' }} }} }}, plugins: {{ legend: {{ display: false }} }} }}
}});

new Chart(document.getElementById('price'), {{
  type: 'doughnut',
  data: {{ labels: ['Confidence', 'Risk'], datasets: [{{ data: [marketPrice * 100, (1 - marketPrice) * 100], backgroundColor: ['#6fdb6f', '#1a3a1a'], borderWidth: 0 }}] }},
  options: {{ cutout: '70%', plugins: {{ legend: {{ labels: {{ color: '#c8e6c9' }} }} }} }}
}});

// Page tab switching
document.querySelectorAll('.page-tab').forEach(btn => {{
  btn.addEventListener('click', () => {{
    document.querySelectorAll('.page-tab').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.page-content').forEach(c => c.classList.remove('active'));
    btn.classList.add('active');
    document.getElementById(btn.dataset.tab + '-tab').classList.add('active');
  }});
}});
</script>
</body></html>
"""
