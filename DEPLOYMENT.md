# Deployment Guide

CancerHawk runs as two services: a **Vercel** static site (frontend) and a **Railway** Docker worker (backend).

## Architecture

```
github.com/asimog/cancerhawk
    ├── pushes to Railway (auto-deploy)
    │   └── Dockerfile → python -m app.main (:8080)
    │       └── Postgres (jobs + blocks)
    │
    └── pushes to Vercel (auto-deploy)
        └── Next.js → cancerhawk.org
            └── API calls → Railway worker
```

## Vercel Frontend

### Setup
1. Go to [vercel.com](https://vercel.com), import `asimog/cancerhawk`
2. Framework: **Next.js**, build: `npm run build`, output: `.next`
3. Add environment variable: `NEXT_PUBLIC_BACKEND_URL=https://cancerhawk-production.up.railway.app`
4. Deploy — auto-deploys on every push to master.

### Verification
```bash
curl https://cancerhawk.org/api/health
# → {"status":"ok","service":"cancerhawk"}
```

## Railway Backend

### Setup
```bash
railway login
railway link --project cancerhawk
railway add --database postgres
railway variables set --service cancerhawk \
  DATABASE_URL='${{Postgres.DATABASE_URL}}' \
  OPENROUTER_API_KEY=sk-or-v1-... \
  GITHUB_TOKEN=ghp_... \
  GITHUB_REPO=asimog/cancerhawk \
  SOLANA_WALLET_ADDRESS=... \
  PAY_SH_SANDBOX=false
railway up
```

### Auto-deploy
Railway auto-deploys on every GitHub push to master. No manual trigger needed.

### Verification
```bash
curl https://cancerhawk-production.up.railway.app/api/health
# → {"status":"ok","service":"cancerhawk"}

curl https://cancerhawk-production.up.railway.app/api/models
# → {"models":[...],"defaults":{...}}

curl -s -X POST https://cancerhawk-production.up.railway.app/api/agents/cost \
  -H "Content-Type: application/json" \
  -d '{"model":"openrouter/free","n_submitters":3}'
```

## Environment Variables

### Railway (backend worker)
| Variable | Required | Purpose |
|---|---|---|
| `DATABASE_URL` | yes | Postgres connection string (Railway auto-links) |
| `OPENROUTER_API_KEY` | yes | OpenRouter key for autonomous runs |
| `GITHUB_TOKEN` | yes | GitHub PAT for block publishing |
| `GITHUB_REPO` | yes | `owner/repo` for git push |
| `SOLANA_WALLET_ADDRESS` | no | Worker wallet for pay.sh |
| `PAY_SH_SANDBOX` | no | `true` (default) sandbox; `false` real |
| `HERMES_AUTO_GENERATE_ENABLED` | no | `true` to auto-generate blocks |
| `MOLTBOOK_API_KEY` | no | Moltbook post key |
| `CANCERHAWK_CORS_ORIGINS` | no | CORS allowlist |
| `GITHUB_BRANCH` | no | Branch to push (default: `master`) |

### Vercel (frontend)
| Variable | Required | Purpose |
|---|---|---|
| `NEXT_PUBLIC_BACKEND_URL` | yes | Railway worker URL |

## Health Checks

| Endpoint | Expected |
|---|---|
| `GET /api/health` | `200 {"status":"ok","service":"cancerhawk"}` |
| `GET /api/healthcheck` | `200` |

## Postgres

Jobs and blocks are persisted to Postgres. Blocks batch-sync to GitHub once daily.

### Tables
- `jobs` — job cards with events, config, result, error
- `blocks` — paper content, analysis, market price, push status

### Fallback
Without `DATABASE_URL`, jobs use an ephemeral `jobs.json` file. This does not survive Railway redeploys.

## Local Development
```bash
pip install -r app/requirements.txt
python -m app.main               # → http://localhost:8765
python -m pytest                 # all tests
npm install && npm run build     # frontend
```
