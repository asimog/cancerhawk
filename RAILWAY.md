# Railway Deployment

CancerHawk worker runs on Railway as a Docker-based service with Postgres.

## One-time Setup
```bash
railway login
railway link --project cancerhawk
railway add --database postgres
railway variables set --service cancerhawk \
  DATABASE_URL='${{Postgres.DATABASE_URL}}' \
  OPENROUTER_API_KEY=sk-or-v1-... \
  GITHUB_TOKEN=ghp_... \
  GITHUB_REPO=asimog/cancerhawk \
  SOLANA_WALLET_ADDRESS=...
```

## Deploy
```bash
railway up                 # manual deploy
# OR push to GitHub — Railway auto-deploys from master
```

## Configuration
```toml
# railway.toml
[build]
builder = "DOCKERFILE"
dockerfilePath = "Dockerfile"

[deploy]
healthcheckPath = "/api/health"
restartPolicyType = "ON_FAILURE"
startCommand = "python -m app.main"
```

## Health Check
Railway polls `/api/health` every 30s. Must return 200 within 5s.

## Postgres
- Automatically linked via `DATABASE_URL` reference variable
- Schema auto-migrated on startup (`app/db.py → init_db()`)
- Blocks batch-sync to GitHub once daily
- Job cards persist across redeploys

## Logs
```bash
railway logs --service cancerhawk
```

## Redeploy
```bash
railway deployment redeploy --service cancerhawk --yes
```

## Environment Variables

| Variable | Required | Default | Purpose |
|---|---|---|---|
| `DATABASE_URL` | yes | – | Postgres connection |
| `OPENROUTER_API_KEY` | yes | – | OpenRouter key |
| `GITHUB_TOKEN` | yes | – | GitHub PAT |
| `GITHUB_REPO` | yes | – | `owner/repo` |
| `SOLANA_WALLET_ADDRESS` | yes | – | Worker wallet for pay.sh |
| `PAY_SH_SANDBOX` | no | `true` | Sandbox/real pay.sh |
| `HERMES_AUTO_GENERATE_ENABLED` | no | – | Auto-generate blocks |
| `MOLTBOOK_API_KEY` | no | – | Moltbook posting |
| `GITHUB_BRANCH` | no | `master` | Git push branch |
| `CANCERHAWK_CORS_ORIGINS` | no | – | CORS allowlist |
