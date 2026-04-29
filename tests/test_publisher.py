"""Unit tests for app.publisher HTML rendering."""

import html
import json

from app import publisher
from app.publisher import (
    _render_peer_reviews,
    _render_simulations,
    _archetype_table,
    _topics_table,
    load_previous_block_context,
    publish_block,
)


def test_render_peer_reviews_empty():
    html_out = _render_peer_reviews([])
    assert "No peer reviews available" in html_out


def test_render_peer_reviews_with_one_review():
    review = {
        "archetype_name": "Oncogene Hunter",
        "recommendation": "accept",
        "confidence": 0.95,
        "summary": "Solid paper with strong evidence.",
        "dimension_scores": {
            "mechanistic_plausibility": 9,
            "experimental_design": 8,
            "evidence_support": 9,
            "statistical_rigor": 8,
            "clarity_of_writing": 7,
        },
        "criticisms": ["Minor typo in section 2."],
        "required_fixes": ["Fix typo."],
        "suggested_experiments": ["Validate in another cell line."],
        "simulation_proposal": {
            "type": "in_silico",
            "description": "Virtual trial with 1000 patients.",
            "rationale": "Confirm efficacy prediction.",
            "expected_metrics": ["response_rate"],
        },
    }
    html_out = _render_peer_reviews([review])

    # Should contain acceptance banner
    assert "Peer review acceptance probability:" in html_out
    assert "100%" in html_out  # accept weight 1.0 * confidence 0.95 / 0.95 = 1

    # Should contain archetype name escaped
    assert "Oncogene Hunter" in html_out

    # Should contain dimension scores (numbers)
    assert "9" in html_out
    assert "8" in html_out

    # `_render_peer_reviews` does NOT render `simulation_proposal` — that
    # field is rendered separately by `_render_simulations`. So neither
    # the proposal type nor description should appear here.
    assert "in_silico" not in html_out
    assert "Virtual trial" not in html_out

    # No raw script tags or unescaped HTML.
    assert "<script>" not in html_out


def test_render_peer_reviews_xss_protection():
    malicious_review = {
        "archetype_name": "<script>alert('xss')</script>",
        "recommendation": "accept",
        "confidence": 0.5,
        "summary": "<img src=x onerror=alert(1)>",
        "dimension_scores": {
            "mechanistic_plausibility": 5,
            "experimental_design": 5,
            "evidence_support": 5,
            "statistical_rigor": 5,
            "clarity_of_writing": 5,
        },
        "criticisms": [],
        "required_fixes": [],
        "suggested_experiments": [],
        "simulation_proposal": None,
    }
    html_out = _render_peer_reviews([malicious_review])
    # The script tag should be escaped
    assert "<script>" not in html_out
    assert "&lt;script&gt;" in html_out
    assert "<img" not in html_out
    assert "&lt;img" in html_out


def test_render_simulations_empty():
    html_out = _render_simulations([])
    assert "No simulation proposals" in html_out


def test_render_simulations_with_entry():
    sims = [
        {
            "id": "bootstrap-response",
            "title": "Bootstrap Response",
            "type": "html5_canvas",
            "description": "Bootstrap resampling of the dataset.",
            "rationale": "Assess robustness to sample variance.",
            "expected_metrics": ["p_value", "confidence_interval"],
            "scene": "counterfactual_perturbation",
            "seed": 123,
        }
    ]
    html_out = _render_simulations(sims)
    assert "html5_canvas" in html_out
    assert "Bootstrap Response" in html_out
    assert "Bootstrap resampling" in html_out
    assert "p_value" in html_out
    assert 'id="sim-bootstrap-response"' in html_out
    assert "simulation-scenes" in html_out
    assert "counterfactual_perturbation" in html_out


def test_render_simulations_includes_threejs_when_present():
    sims = [
        {
            "id": "canvas-2d",
            "title": "Canvas 2D",
            "type": "html5_canvas",
            "description": "2D scene.",
            "rationale": "fast",
            "expected_metrics": ["x"],
            "scene": "trajectory_manifold",
            "seed": 1,
        },
        {
            "id": "tumor-volume-3d",
            "title": "Tumor Volume 3D",
            "type": "threejs",
            "description": "3D volumetric scene.",
            "rationale": "depth",
            "expected_metrics": ["volumetric_stability"],
            "three_scene": "tumor_volume_3d",
            "seed": 2,
            "parameters": {"confidence": 0.6},
        },
    ]
    out = _render_simulations(sims)
    assert '<canvas id="sim-canvas-2d"' in out
    assert 'id="three-tumor-volume-3d"' in out
    assert "three-stage" in out
    assert 'type="importmap"' in out
    assert "three.module.js" in out
    assert "HTML5 Canvas (2D)" in out
    assert "Three.js (3D / WebGL)" in out


def test_render_simulations_omits_threejs_when_absent():
    sims = [
        {
            "id": "only-2d",
            "title": "Only 2D",
            "type": "html5_canvas",
            "description": "2D scene.",
            "rationale": "fast",
            "expected_metrics": ["x"],
            "scene": "trajectory_manifold",
            "seed": 1,
        }
    ]
    out = _render_simulations(sims)
    assert '<canvas id="sim-only-2d"' in out
    # Three.js plumbing must not be injected when no 3D specs are present.
    assert 'type="importmap"' not in out
    assert "three.module.js" not in out
    assert "three-stage" not in out
    assert "Three.js (3D / WebGL)" not in out


def test_render_simulations_escapes_script_payload():
    sims = [
        {
            "id": "bad",
            "title": "</script><script>alert(1)</script>",
            "description": "<img src=x onerror=alert(1)>",
            "rationale": "safe",
            "expected_metrics": [],
            "scene": "trajectory_manifold",
        }
    ]
    html_out = _render_simulations(sims)
    assert "</script><script>" not in html_out
    assert "<img" not in html_out
    assert "\\u003c/script" in html_out


def test_archetype_table_renders():
    archetypes = [
        {
            "archetype_name": "Test Archetype",
            "scores": {
                "clinical_viability": 8,
                "regulatory_risk": 3,
                "market_potential": 7,
                "patient_impact": 9,
                "novelty": 6,
                "falsifiability": 5,
            },
            "verdict": "Promising but needs more validation.",
        }
    ]
    html_out = _archetype_table(archetypes)
    assert "Test Archetype" in html_out
    assert "8" in html_out  # clinical viability
    assert "Promising but needs more validation" in html_out


def test_archetype_table_escapes_non_numeric_scores():
    archetypes = [
        {
            "archetype_name": "Injected",
            "scores": {
                "clinical_viability": "<img src=x onerror=alert(1)>",
            },
            "verdict": "<script>alert(1)</script>",
        }
    ]
    html_out = _archetype_table(archetypes)
    assert "<img" not in html_out
    assert "<script>" not in html_out
    assert "&lt;img" in html_out
    assert "&lt;script&gt;" in html_out


def test_topics_table_renders():
    topics = [
        {
            "id": "T1",
            "title": "Next block topic",
            "probability": 0.75,
            "impact": 9,
            "token_cost": 5000,
            "rationale": "Because it follows.",
        }
    ]
    html_out = _topics_table(topics)
    assert "T1" in html_out
    assert "Next block topic" in html_out
    assert "0.75" in html_out


def test_topics_table_escapes_all_cells():
    topics = [
        {
            "id": "<script>1</script>",
            "title": "<img src=x>",
            "probability": "<b>bad</b>",
            "impact": "<svg>",
            "token_cost": "<iframe>",
            "rationale": "<script>2</script>",
        }
    ]
    html_out = _topics_table(topics)
    assert "<script>" not in html_out
    assert "<img" not in html_out
    assert "<b>" not in html_out
    assert "<svg>" not in html_out
    assert "<iframe>" not in html_out
    assert "&lt;script&gt;" in html_out


def test_publish_block_rewrites_index_to_latest_block(tmp_path, monkeypatch):
    """The Pages root is `results/index.html`, so every publish must rewrite
    it with the newest block rather than leaving an older result visible.
    """

    class FakePaper:
        title = "Fresh CancerHawk Block"
        sections = [{"heading": "Finding", "content": "Latest block content."}]
        accepted_submissions = ["direction"]
        rejections = []

        def full_text(self):
            return "# Fresh CancerHawk Block\n\n## Finding\n\nLatest block content."

    class FakeAnalysis:
        archetypes = []
        market_price = 0.61
        score_matrix = {}
        consensus_dim = {}
        headline_catalysts = []

    monkeypatch.setattr(publisher, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(publisher, "RESULTS_DIR", tmp_path / "results")

    meta_1 = publish_block(
        paper=FakePaper(),
        analysis=FakeAnalysis(),
        derived_topics=[],
        research_goal="first goal",
        models={},
        peer_reviews=[],
        simulations=[],
    )
    meta_2 = publish_block(
        paper=FakePaper(),
        analysis=FakeAnalysis(),
        derived_topics=[],
        research_goal="second goal",
        models={},
        peer_reviews=[],
        simulations=[],
    )

    index_html = (tmp_path / "results" / "index.html").read_text(encoding="utf-8")
    archive_html = (tmp_path / "results" / "blocks.html").read_text(encoding="utf-8")
    assert meta_1["block"] == 1
    assert meta_2["block"] == 2
    assert "CancerHawk · Block 2" in index_html
    assert "second goal" in index_html
    assert "block-2/paper.html" in index_html
    assert "CancerHawk Block Archive" in archive_html
    assert "block-1/paper.html" in archive_html
    assert "block-2/paper.html" in archive_html


def test_load_previous_block_context_includes_citable_urls(tmp_path, monkeypatch):
    monkeypatch.setattr(publisher, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(publisher, "RESULTS_DIR", tmp_path / "results")

    class FakePaper:
        title = "Prior Mechanism"
        sections = [{"heading": "Mechanism", "content": "A useful prior finding."}]
        accepted_submissions = ["direction"]
        rejections = []

        def full_text(self):
            return "# Prior Mechanism\n\n## Mechanism\n\nA useful prior finding."

    class FakeAnalysis:
        archetypes = []
        market_price = 0.5
        score_matrix = {}
        consensus_dim = {}
        headline_catalysts = []

    publish_block(
        paper=FakePaper(),
        analysis=FakeAnalysis(),
        derived_topics=[{"title": "Follow the signal", "rationale": "It extends the prior."}],
        research_goal="prior goal",
        models={},
        peer_reviews=[],
        simulations=[],
    )

    context = load_previous_block_context()
    assert "CancerHawk Block 1" in context
    assert "https://asimog.github.io/cancerhawk/block-1/paper.html" in context
    assert "A useful prior finding" in context
