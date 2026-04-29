# CancerHawk — MOTO-Style UX + Peer Review

## What's New

### 1. MOTO-Style Real-Time UI (in browser)
- **Eternal sidebar** (right side, collapsible) shows:
  - Current pipeline phase badge
  - Research timer (HH:MM:SS)
  - Token totals (input/output/total) with color coding
  - Per-model breakdown (expandable)
  - API call stats (success/fail rate)
- **Stats dashboard** (6 metric cards): calls, tokens, phase, success rate, latency, cost
- **Enhanced API log table**:
  - Filterable by role and status
  - Expandable rows show full prompt + response
  - Copy-to-clipboard buttons
  - Summary stats row

### 2. Structured Server Logging
Backend logs now use millisecond-precision timestamps:
```
2026-04-28 05:03:59.641 | INFO  | api_call seq=1 role=submitter model=anthropic/claude-haiku-4.5 ...
2026-04-28 05:04:01.123 | INFO  | stage_start stage=paper_engine
2026-04-28 05:05:30.456 | INFO  | run_complete title="..." total_calls=42 total_tokens=12345 ...
```

### 3. Peer Review by MiroShark Archetypes
After analysis completes, all 8 archetype agents independently review the paper:
- **Recommendation**: accept / minor_revision / major_revision / reject
- **Dimension scores**: mechanistic plausibility, experimental design, evidence support, statistical rigor, clarity
- **Criticisms**, **required fixes**, **suggested experiments**
- **Simulation proposal**: each reviewer suggests a computational/statistical test

The system synthesizes:
- Acceptance probability (weighted consensus)
- Major concerns (cross-cutting issues)
- Recommended simulations (top 3 prioritized)
- Revision priorities

### 4. 3-Page Published Paper
Every block in `results/block-N/paper.html` now has **three tabs**:

1. **Paper** — original content, charts, archetype scores, topics
2. **Peer Reviews** — all 8 reviews with expandable cards, acceptance banner
3. **Simulations** — recommended computational experiments

The `results/index.html` landing page shows the latest block with all three sections embedded.

## Quick Start

### Windows
Double-click **`install_cancerhawk.bat`** once, then **`run_cancerhawk.bat`** to start the worker. The run window shows live API call logs.

### Manual
```bash
cd D:\mythOS\cancerhawk
pip install -r app/requirements.txt
python -m app.main
```
Open <http://localhost:8765>. The FastAPI backend serves both the API and the static UI from `app/web/` — there is no separate frontend dev server.

## Architecture

```
main.py
  ├─ run_paper_engine()        # brainstorm → validate → compile
  ├─ run_analysis_engine()     # 8 archetypes score paper
  ├─ run_peer_review_engine()  # 8 archetypes review as peer reviewers
  ├─ topic_deriver_prompt()    # next-block topics
  └─ publish_block()           # writes results/block-N/

peer_review_engine.py
  ├─ _review_one()   — per-archetype review via LLM
  └─ _synthesize()   — aggregate → acceptance prob, sim priorities

publisher.py
  ├─ _render_peer_reviews()   — HTML cards per review + banner
  ├─ _render_simulations()    — simulation proposal cards
  └─ _PAGE_SHELL              — 3-tab layout + tab switching JS

app/web/
  ├─ index.html   — stats dashboard, sidebar, calls table
  ├─ app.js       — real-time WS updates, expandable rows, filters
  └─ styles.css   — MOTO dark theme + responsive layout
```

## Token Tracking

Every LLM call is recorded with:
- `seq` — sequential number
- `role` — which pipeline component made the call (submitter, validator, compiler, archetype, topic_deriver, peer_review:oncologist, …)
- `model`, `prompt_tokens`, `completion_tokens`, `latency_ms`, `cost_usd`, `ok`, `error`

Stats are aggregated in `TokenTracker.stats()` and sent to the UI via WebSocket on every call.

## File Output

Each `results/block-N/` contains:
- `paper.md` — raw markdown
- `paper.html` — 3-tab page (paper | peer reviews | simulations)
- `analysis.json` — archetype scores + peer_reviews[] + simulations[]
- `block.json` — metadata + `has_peer_review`, `has_simulations` flags

## Batch Files

| File | Purpose |
|------|---------|
| **install_cancerhawk.bat** | One-time setup — checks Python, upgrades pip, installs `app/requirements.txt`, creates `results/` |
| **run_cancerhawk.bat** | Daily launcher — frees port 8765 if stale, opens browser, runs the FastAPI worker in the foreground with live API call logs |

The backend window displays the same structured log format as MOTO.
