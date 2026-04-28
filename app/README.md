# CancerHawk Engine

A self-contained, browser-driven autonomous oncology research engine.
Combines a **full MOTO** paper compiler (adaptive, indefinite aggregation —
not the simplified 3-round / 3-accept variant) with a MiroShark-style
multi-archetype analysis and peer-review layer. All LLM calls go through
OpenRouter using a key you paste into the browser — nothing is stored
server-side.

## Pipeline (one click → one block)

```
brainstorm  →  validate  →  compile paper  →  archetype analysis  →  peer review
   N parallel    accept/    section-by-       8 archetypes score      8 archetypes
   submitters    reject     section by        on 6 dimensions         review paper
   per round     + steer    compiler                                  + simulations
   (loops with   + aggregate
    convergence
    detection)
                                              ↓
                                          synthesis market price
                                              ↓
                                          derive next-block topics
                                              ↓
                                          publish to results/block-N/
                                              ↓
                                          (optional) git push → Pages
```

## Run

```bash
pip install -r app/requirements.txt
python app/main.py
```

Open <http://localhost:8765>, paste your OpenRouter API key, type a research
goal, pick models, click run. The browser shows live progress; the finished
block lands in `results/block-N/` and `results/index.html` is rewritten to
show the latest one.

## Files

| File | Purpose |
|---|---|
| `main.py` | FastAPI app — UI + WebSocket pipeline orchestrator |
| `openrouter.py` | Async OpenRouter chat client (key per-call) |
| `prompts.py` | All system prompts and the 8 archetype definitions |
| `paper_engine.py` | Brainstorm → validate → compile loop |
| `analysis_engine.py` | 8 archetype agents + synthesis-market price |
| `publisher.py` | Writes `results/block-N/`, rewrites `results/index.html` |
| `web/index.html`, `app.js`, `styles.css` | Single-page UI |

## Output layout

```
results/
  index.html               ← always shows the latest block (auto-rewritten)
  block-1/
    paper.md               ← raw paper markdown
    paper.html             ← standalone block page with charts
    analysis.json          ← full archetype scores + market price + topics
    block.json             ← run metadata
  block-2/
    ...
```

`results/index.html` is what GitHub Pages serves — the existing
`.github/workflows/static.yml` already publishes the whole repo on push.
Tick the **also git add/commit/push** box in the UI and every block goes
live to Pages automatically.

## Customizing

- **Add models to the dropdown** — edit `MODELS` in `main.py`.
- **Change defaults** — edit `DEFAULT_MODELS` in `main.py`.
- **Re-tune prompts** — `prompts.py` has all role prompts and the 8
  archetype definitions in one file.
- **Tune adaptive convergence** — full MOTO has *no* fixed round/accept
  cap. Behavior is controlled by environment variables read in
  `paper_engine.py`:
  - `CANCERHAWK_MIN_ACCEPTED` (default 3) — minimum accepted submissions
    before the convergence detector is allowed to stop.
  - `CANCERHAWK_SATURATION_ROUNDS` (default 2) — stop after K consecutive
    rounds with zero acceptances.
  - `CANCERHAWK_PLATEAU_ROUNDS` (default 3) — stop when avg validator
    novelty score is non-increasing for K rounds.
  - `CANCERHAWK_MAX_CALLS` (default 400) — soft safety guard on total API
    calls. Set to 0 to disable.
  - `CANCERHAWK_MAX_WALL_CLOCK` (default 3600s) — soft safety guard on
    wall-clock seconds. Set to 0 to disable.
- **Change submitters per round** — `n_submitters` in the WebSocket config
  (UI input, 1–8).

## Privacy

The OpenRouter API key lives in your browser only. It is sent to the local
FastAPI server per WebSocket session, used to call `openrouter.ai`, and
discarded. The "remember in this browser" checkbox stores it in
`localStorage` — clear it via browser devtools if you change your mind.
