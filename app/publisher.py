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


def _rewrite_index() -> None:
    blocks = []
    for entry in RESULTS_DIR.iterdir():
        m = BLOCK_DIR_RE.match(entry.name)
        if not m:
            continue
        meta_path = entry / "block.json"
        analysis_path = entry / "analysis.json"
        if not meta_path.is_file() or not analysis_path.is_file():
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
        paper_md = (entry / "paper.md").read_text(encoding="utf-8")
        blocks.append((int(m.group(1)), meta, analysis, paper_md))

    blocks.sort(key=lambda x: x[0], reverse=True)
    if blocks:
        latest = blocks[0]
        index_html = _render_index(latest, blocks)
    else:
        index_html = _empty_index()
    (RESULTS_DIR / "index.html").write_text(index_html, encoding="utf-8")


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
        sim_type = html.escape(str(s.get("type", "threejs_html5")))
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
            f'<canvas id="sim-{sim_id}" aria-label="{title} interactive Three.js simulation"></canvas>'
            f'<div class="simulation-overlay"><span>Native Three.js</span><strong>{title}</strong></div>'
            f'</div>'
            f'</div>'
        )

    return (
        '<div class="simulation-intro">'
        '<p>Runnable browser-native simulations generated after peer review. '
        'Each scene uses HTML5 canvas plus Three.js to turn the paper into a falsifiable visual model.</p>'
        '</div>'
        + "".join(sims_html)
        + _simulation_script(scene_payloads)
    )


def _render_block_page(paper, analysis_payload: dict, meta: dict) -> str:
    sections_html = "\n".join(
        f'<section><h2>{html.escape(s["heading"])}</h2>'
        f'<div class="prose">{_md_inline_to_html(s["content"])}</div></section>'
        for s in paper.sections
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
        analysis_json=html.escape(json.dumps(analysis_payload, indent=2)),
        consensus_json=html.escape(json.dumps(analysis_payload["consensus_dim"])),
        score_matrix_json=html.escape(json.dumps(analysis_payload["score_matrix"])),
        block_history_html="",
    )


def _render_index(latest, all_blocks) -> str:
    block_n, meta, analysis_payload, paper_md = latest
    # Reconstruct paper sections from markdown for display
    sections_html = _md_to_sections_html(paper_md)
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
        analysis_json=html.escape(json.dumps(analysis_payload, indent=2)),
        consensus_json=html.escape(json.dumps(analysis_payload["consensus_dim"])),
        score_matrix_json=html.escape(json.dumps(analysis_payload["score_matrix"])),
        block_history_html=history_html,
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
<script type="module">
import * as THREE from 'https://cdn.jsdelivr.net/npm/three@0.164.1/build/three.module.js';

const scenes = JSON.parse(document.getElementById('simulation-scenes')?.textContent || '[]');
const palette = [0x6fdb6f, 0x42c6ff, 0xffc857, 0xff6b6b, 0xb892ff];

function seededRandom(seed) {{
  let value = Math.max(1, seed || 1) % 2147483647;
  return () => {{
    value = value * 16807 % 2147483647;
    return (value - 1) / 2147483646;
  }};
}}

function createRenderer(canvas) {{
  const renderer = new THREE.WebGLRenderer({{ canvas, antialias: true, alpha: true }});
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  return renderer;
}}

function resize(renderer, camera, canvas) {{
  const rect = canvas.parentElement.getBoundingClientRect();
  const width = Math.max(320, Math.floor(rect.width));
  const height = Math.max(280, Math.floor(rect.height));
  if (canvas.width !== width || canvas.height !== height) {{
    renderer.setSize(width, height, false);
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
  }}
}}

function addLights(scene) {{
  scene.add(new THREE.AmbientLight(0xbdf9bd, 0.72));
  const key = new THREE.PointLight(0x6fdb6f, 2.4, 80);
  key.position.set(8, 10, 10);
  scene.add(key);
  const rim = new THREE.PointLight(0x42c6ff, 1.6, 70);
  rim.position.set(-10, -5, -8);
  scene.add(rim);
}}

function makeCell(color, size = 0.22) {{
  const geo = new THREE.SphereGeometry(size, 24, 16);
  const mat = new THREE.MeshStandardMaterial({{
    color,
    emissive: color,
    emissiveIntensity: 0.22,
    roughness: 0.38,
    metalness: 0.08,
    transparent: true,
    opacity: 0.9
  }});
  return new THREE.Mesh(geo, mat);
}}

function trajectoryManifold(scene, rand) {{
  const group = new THREE.Group();
  for (let strand = 0; strand < 5; strand++) {{
    const points = [];
    const color = palette[strand % palette.length];
    for (let i = 0; i < 52; i++) {{
      const t = i / 8;
      points.push(new THREE.Vector3(
        Math.sin(t + strand) * (1.2 + strand * 0.22) + (rand() - 0.5) * 0.18,
        Math.cos(t * 0.8 + strand * 0.4) * 0.9 + strand * 0.25 - 0.5,
        (i - 26) * 0.09 + Math.sin(t * 0.5) * 0.5
      ));
    }}
    const curve = new THREE.CatmullRomCurve3(points);
    const tube = new THREE.Mesh(
      new THREE.TubeGeometry(curve, 110, 0.025, 8, false),
      new THREE.MeshStandardMaterial({{ color, emissive: color, emissiveIntensity: 0.55 }})
    );
    group.add(tube);
    for (let i = 0; i < points.length; i += 8) {{
      const cell = makeCell(color, 0.12 + rand() * 0.06);
      cell.position.copy(points[i]);
      group.add(cell);
    }}
  }}
  scene.add(group);
  return (time) => {{
    group.rotation.y = time * 0.18;
    group.rotation.x = Math.sin(time * 0.3) * 0.14;
  }};
}}

function counterfactualPerturbation(scene, rand) {{
  const group = new THREE.Group();
  const control = new THREE.Group();
  const treated = new THREE.Group();
  for (let i = 0; i < 34; i++) {{
    const angle = i * 0.34;
    const c = makeCell(0x42c6ff, 0.12);
    c.position.set(Math.cos(angle) * 1.1 - 1.45, Math.sin(angle * 1.2) * 0.8, (i - 17) * 0.08);
    control.add(c);
    const t = makeCell(i > 14 ? 0xff6b6b : 0x6fdb6f, 0.12);
    t.position.set(Math.cos(angle) * (1.1 + i * 0.025) + 1.35, Math.sin(angle * 1.45) * (0.8 + i * 0.01), (i - 17) * 0.08);
    treated.add(t);
  }}
  const pulse = new THREE.Mesh(
    new THREE.TorusGeometry(1.35, 0.025, 12, 96),
    new THREE.MeshStandardMaterial({{ color: 0xffc857, emissive: 0xffc857, emissiveIntensity: 0.9 }})
  );
  pulse.position.x = 1.35;
  pulse.rotation.x = Math.PI / 2;
  group.add(control, treated, pulse);
  scene.add(group);
  return (time) => {{
    control.rotation.y = time * 0.22;
    treated.rotation.y = -time * 0.22;
    pulse.scale.setScalar(1 + Math.sin(time * 2.4) * 0.18);
    pulse.material.opacity = 0.7 + Math.sin(time * 2.4) * 0.2;
  }};
}}

function microenvironmentGradient(scene, rand) {{
  const group = new THREE.Group();
  const plane = new THREE.Mesh(
    new THREE.PlaneGeometry(6, 3.6, 48, 28),
    new THREE.MeshBasicMaterial({{ color: 0x103510, transparent: true, opacity: 0.34, wireframe: true }})
  );
  plane.rotation.x = -Math.PI / 2;
  group.add(plane);
  for (let i = 0; i < 70; i++) {{
    const stress = rand();
    const cell = makeCell(stress > 0.62 ? 0xff6b6b : stress > 0.35 ? 0xffc857 : 0x6fdb6f, 0.08 + rand() * 0.08);
    cell.position.set((rand() - 0.5) * 5.4, (stress - 0.5) * 1.2, (rand() - 0.5) * 3.0);
    cell.userData.phase = rand() * Math.PI * 2;
    group.add(cell);
  }}
  scene.add(group);
  return (time) => {{
    group.rotation.y = Math.sin(time * 0.25) * 0.35;
    group.children.forEach((child) => {{
      if (child.userData.phase !== undefined) {{
        child.position.y += Math.sin(time * 1.4 + child.userData.phase) * 0.002;
      }}
    }});
  }};
}}

function bootScene(spec) {{
  const canvas = document.getElementById(`sim-${{spec.id}}`);
  if (!canvas) return;
  const renderer = createRenderer(canvas);
  const scene = new THREE.Scene();
  scene.fog = new THREE.FogExp2(0x050a06, 0.055);
  const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 100);
  camera.position.set(0, 1.6, 7.2);
  addLights(scene);
  const rand = seededRandom(spec.seed);
  const animateScene = spec.scene === 'counterfactual_perturbation'
    ? counterfactualPerturbation(scene, rand)
    : spec.scene === 'microenvironment_gradient'
      ? microenvironmentGradient(scene, rand)
      : trajectoryManifold(scene, rand);
  function frame(ms) {{
    const time = ms / 1000;
    resize(renderer, camera, canvas);
    animateScene(time);
    renderer.render(scene, camera);
    requestAnimationFrame(frame);
  }}
  requestAnimationFrame(frame);
}}

scenes.forEach(bootScene);
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


def _md_inline_to_html(text: str) -> str:
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    return "\n".join(f"<p>{html.escape(p)}</p>" for p in paras)


def _md_to_sections_html(md: str) -> str:
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
.simulation-stage {{ position: relative; min-height: 320px; border-radius: 14px; overflow: hidden; background: radial-gradient(circle at 50% 40%, rgba(111,219,111,0.18), rgba(5,10,6,0.96) 62%); border: 1px solid rgba(111,219,111,0.24); }}
.simulation-stage canvas {{ width: 100%; height: 100%; display: block; }}
.simulation-overlay {{ position: absolute; left: 14px; right: 14px; bottom: 12px; display: flex; justify-content: space-between; gap: 10px; align-items: center; color: #d7ffd7; font-size: 12px; text-shadow: 0 1px 8px #000; pointer-events: none; }}
.simulation-overlay span {{ color: #42c6ff; text-transform: uppercase; letter-spacing: 0.08em; }}
.simulation-overlay strong {{ text-align: right; max-width: 60%; }}
@media (max-width: 860px) {{ .simulation-card {{ grid-template-columns: 1fr; }} .simulation-stage {{ min-height: 280px; }} }}
</style>
</head><body>
<header>
  <h1>{title}</h1>
  <p class="meta">CancerHawk · Block {block} · {timestamp}</p>
  <p class="meta"><strong>Research goal:</strong> {research_goal}</p>
</header>

<aside class="disclaimer">
  Autonomously generated by CancerHawk. This paper has undergone automated peer review by MiroShark archetype agents. May contain incorrect, incomplete, or fabricated claims. Independently verify before acting on any content.
</aside>

<div class="market-banner">
  <div><div class="label">Synthesis market price</div><div class="price">{market_pct}%</div></div>
  <div><div class="label">Verdict</div><div>Aggregated archetype confidence in clinical + commercial viability</div></div>
</div>

<!-- Page Tabs -->
<div class="page-tabs">
  <button class="page-tab active" data-tab="paper">Paper</button>
  <button class="page-tab" data-tab="peer-reviews">Peer Reviews</button>
  <button class="page-tab" data-tab="simulations">Simulations</button>
</div>

<!-- Paper Content -->
<div id="paper-tab" class="page-content active">
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
