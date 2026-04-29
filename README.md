# CancerHawk

Autonomous oncology research blocks. Each "block" is a peer-reviewed paper plus
2D and 3D simulations, generated end-to-end by:

- **Full MOTO paper engine** — adaptive, indefinite aggregation of research
  directions (no fixed round/accept caps; converges via saturation + novelty
  plateau).
- **MiroShark peer review** — 8 archetype agents (oncologist, biostatistician,
  FDA, investor, KOL, patient, payer, short-seller) score the paper on five
  dimensions and propose simulations.
- **Simulation engine** — emits two visualization tracks per block: HTML5
  Canvas (2D) scenes for fast falsifier views and Three.js (WebGL) 3D scenes
  for volumetric tumor / mitotic / perturbation views.
- **Publisher** — writes `results/block-N/{paper.md,paper.html,analysis.json,
  block.json}` and rewrites `results/index.html` so the public site always
  shows the latest block.

## Architecture

```text
┌───────────────────────────────────┐
│  Vercel site (cancerhawk site)    │
│  meetsurveyman account            │
│  - serves results/* statically    │
│  - run UI at /run.html            │
└────┬──────────────────────────────┘
     │ wss:// (WebSocket /ws/run)
     ▼
┌───────────────────────────────────┐
│  Railway worker                   │
│  Project aaf250a7-...             │
│  - app/main.py FastAPI            │
│  - MOTO + peer review + sims      │
└────┬──────────────────────────────┘
     │ git push (GITHUB_TOKEN)
     ▼
┌───────────────────────────────────┐
│  GitHub: asimog/cancerhawk        │
│  master branch                    │
│  - Vercel auto-rebuilds on push   │
│  - GH Pages mirrors as fallback   │
└───────────────────────────────────┘
```

Per-block flow: user opens the Vercel site → pastes OpenRouter key → clicks
Run → Vercel UI opens a WebSocket to the Railway worker → worker runs the
pipeline, streams progress → on completion the worker commits
`results/block-N/` to GitHub → Vercel rebuilds → new block appears publicly.

## Local development

Windows:

```cmd
install_cancerhawk.bat
run_cancerhawk.bat
```

Other platforms:

```bash
pip install -r app/requirements.txt
python -m app.main
# open http://localhost:8765
```

The full engine internals are documented in [app/README.md](app/README.md).

## Environment variables

### Worker (Railway / local backend)

| Variable | Default | Purpose |
|---|---|---|
| `PORT` | `8765` | HTTP/WebSocket port. Railway injects this automatically. |
| `CANCERHAWK_CORS_ORIGINS` | GH Pages + localhost | Comma-separated allow-list of origins permitted to call the worker. |
| `CANCERHAWK_PUBLIC_BASE_URL` | `https://asimog.github.io/cancerhawk` | Public URL embedded in generated HTML for absolute links. |
| `CANCERHAWK_BACKEND_URL` | `http://localhost:8765` | Backend URL embedded in `results/run.html` so the static run page knows where to connect. Set this to the Railway public domain. |
| `GITHUB_TOKEN` | _(unset)_ | GitHub PAT with `repo` scope. When set, the worker pushes new blocks back to GitHub using a token-authenticated remote. |
| `GITHUB_REPO` | _(unset)_ | `owner/repo` form, e.g. `asimog/cancerhawk`. Required alongside `GITHUB_TOKEN` for worker-mode push. |
| `GITHUB_BRANCH` | `master` | Branch to push to. |
| `GIT_COMMITTER_NAME` | `cancerhawk-worker` | Committer identity for worker-mode pushes. |
| `GIT_COMMITTER_EMAIL` | `worker@cancerhawk.local` | Committer email. |

### Adaptive convergence (full MOTO)

| Variable | Default | Purpose |
|---|---|---|
| `CANCERHAWK_MIN_ACCEPTED` | `3` | Minimum accepted submissions before the convergence detector is allowed to stop. |
| `CANCERHAWK_SATURATION_ROUNDS` | `2` | Stop when this many consecutive rounds yield zero acceptances. |
| `CANCERHAWK_PLATEAU_ROUNDS` | `3` | Stop when avg validator-novelty score is non-increasing for this many rounds. |
| `CANCERHAWK_MAX_CALLS` | `400` | Soft safety guard on total API calls (set `0` to disable). |
| `CANCERHAWK_MAX_WALL_CLOCK` | `3600` | Soft safety guard on wall-clock seconds (set `0` to disable). |

There are no hard round/accept caps — full MOTO runs as long as the field has
signal under the supplied research goal.

## Deployment

See `railway.json` + `nixpacks.toml` + `Procfile` for Railway, and
`vercel.json` + `.vercelignore` for Vercel. After committing changes, deploy
with:

```bash
# Railway worker
npm i -g @railway/cli
railway login
railway link aaf250a7-c2e0-452c-8546-c1e4b51a8ac4
railway variables set GITHUB_TOKEN=... GITHUB_REPO=asimog/cancerhawk \
  CANCERHAWK_PUBLIC_BASE_URL=https://cancerhawk.vercel.app \
  CANCERHAWK_BACKEND_URL=https://<railway-domain>
railway up
railway domain    # provision public URL

# Vercel site
npm i -g vercel
vercel login              # meetsurveyman account
vercel link --project cancerhawk
vercel deploy --prod
```

Vercel can also be connected to the GitHub repo via the Vercel dashboard for
auto-deploys on every push (recommended).

## Repo layout

```text
app/                  Engine: paper, analysis, peer review, simulations, publisher
results/              Published blocks (block-N/) + index.html (latest)
tests/                115 unit + integration + regression tests
hermes-agent/         Vendored Nous Research Hermes Agent (reference)
MiroShark/, G0DM0D3/  Vendored upstream codebases (reference)
```

## Testing

```bash
python -m pytest        # 115 tests, ~1.3s
```

Code published to <https://github.com/asimog/cancerhawk>.
