# Testing

## Quick Start
```bash
pip install -r app/requirements.txt pytest pytest-asyncio nacl
python -m pytest -q
```

## Test Coverage

| File | Count | What it covers |
|---|---|---|
| `tests/test_health.py` | 4 | `/api/health`, `/api/healthcheck` — status 200, JSON content type |
| `tests/test_agents_api.py` | 24 | `/api/agents/prompts`, `/cost`, `/submit`, `/run`, `/leaderboard`, `/models`, `/enrich` |
| `tests/test_agents.py` | 18 | `parse_agent_run`, `submit_paper`, `get_leaderboard`, cost estimation |
| `tests/test_api.py` | 6 | Root HTML endpoint, models, block 404 |
| `tests/test_security.py` | 10 | Secrets in .env.example, length limits, publisher path safety, medical guidance |
| `tests/test_jobs.py` | 26 | JSON job store roundtrip, status updates, events, idempotency |
| `tests/test_db.py` | 9 | Postgres create/get/update/list jobs, save/get/list blocks, unpushed sync (requires DATABASE_URL) |
| `tests/test_openrouter.py` | 16 | JSON extraction, error handling, retry logic, response parsing |
| `tests/test_extract_json.py` | 4 | JSON extraction edge cases |
| `tests/test_publisher.py` | 8 | Block HTML generation, archetype table rendering |
| `tests/test_peer_review.py` | 3 | Peer review consolidation |
| `tests/test_peer_review_synthesize.py` | 4 | Synthesis engine |
| `tests/test_simulation_engine.py` | 7 | HTML5 simulation generation |
| `tests/test_token_tracker.py` | 10 | Token counting, pricing, limits |
| `tests/test_deployment_config.py` | 5 | Dockerfile, vercel.json, railway.json existence |
| `tests/test_integration.py` | 6 | WebSocket run lifecycle |
| `tests/test_run_job_e2e.py` | 5 | Job creation, idempotency, config persistence |
| `tests/test_homepage_assets.py` | 2 | Static asset serving |
| `tests/test_paper_engine.py` | 10 | MOTO paper engine |

## Running Specific Suites
```bash
python -m pytest tests/test_health.py -v
python -m pytest tests/test_security.py -v
python -m pytest tests/test_db.py -v          # requires DATABASE_URL
python -m pytest tests/ -q --ignore=tests/test_db.py
```

## CI
```yaml
# .github/workflows/ci.yml
backend:
  - pip install -r app/requirements.txt pytest pytest-asyncio nacl
  - python -m pytest -q --tb=short
  - python scripts/scan_secrets.py
  - python scripts/verify_backend.py

frontend:
  - npm install
  - npm run lint
  - npm run build

docker:
  - docker build -t cancerhawk-worker .
```
