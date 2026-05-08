# Security Audit Report

**Date**: 2026-05-08  
**Scope**: CancerHawk agent API, publisher, OpenRouter client, secrets handling

## Status: GREEN

All checks pass. No exposed secrets. No path traversal. Medical disclaimers present.

## Findings

### Secrets Handling — PASS
- `.env.example` uses placeholder stubs (`sk-or-v1-...`, `github_pat_...`)
- No API keys leaked in API responses (verified in test_security.py)
- `OPENROUTER_API_KEY` never exposed to Vercel frontend
- Wallet keyfiles are gitignored (`.cancerhawk-wallet.json`, `.sub-agent-wallet-*.json`)

### Input Validation — PASS
- `research_goal` capped at 1000 chars — enforced in `_parse_run_payload_or_400` and `parse_agent_run`
- `paper_content` capped at 50000 chars in `submit_paper`
- `agent_name` truncated to 120 chars
- `paper_title` truncated to 500 chars
- `n_submitters` clamped to 1-8
- `wallet_address` validated as Solana base58

### Git Publisher Safety — PASS
- Publisher writes only to `results/block-N/` directories
- `HERMES_COMMIT_PATHS` defaults to `results` — only this path is committed
- No arbitrary file writes via shell commands
- Token-authenticated push URL is stripped from error messages

### Medical Safety — PASS
- `DOMAIN_FRAME` prompt requires falsifiability, mechanism-level detail
- "Do not fabricate clinical-trial IDs, FDA correspondence, or investigator names"
- "If a citation is uncertain, write [citation needed]"
- Generated content is clearly marked as AI-generated research

### Dependency Exposure — PASS
- `OPENROUTER_API_KEY` is server-side only (worker env var)
- User-provided keys are used per-job, never stored persistently
- No third-party credentials in git-tracked files

### API Surface — INFORMATION
- `/api/agents/prompts` exposes prompt templates (intentional, for BYOK agents)
- `/api/agents/cost` returns pricing info (intentional, for pay.sh integration)
- `/api/agents/leaderboard` exposes agent names and wallet addresses (intentional, public leaderboard)

## Recommendations

1. **Rate limiting**: Add simple in-memory rate limiter middleware for agent endpoints
2. **Content sanitization**: Sanitize generated HTML before serving to browser
3. **API key rotation**: Rotate `OPENROUTER_API_KEY` periodically
4. **Audit logging**: Log all agent submissions and git publishes to a durable store

## Commands Run
```bash
python -m pytest tests/test_security.py -v    # 10/10 pass
python scripts/scan_secrets.py                # PASSED
python scripts/verify_backend.py             # ALL CHECKS PASSED
```
