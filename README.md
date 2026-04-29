# CancerHawk

Autonomous oncology research blocks powered by a paper engine, archetype
analysis, peer review, and static publishing.

## Run Locally

```bash
uv run --with-requirements app/requirements.txt --with-requirements app/requirements-dev.txt python -m app.main
```

Open `http://localhost:8765`, paste an OpenRouter key, choose models, and run a
fresh block. The app writes each completed block to `results/block-N/` and
rewrites `results/index.html` so the latest block is always the displayed page.

The public Pages site also includes `results/run.html`, a static launcher that
detects and embeds the local backend when `http://localhost:8765` is running.
GitHub Pages cannot run the Python backend by itself; users start the backend
locally, provide their OpenRouter API key there, and generate a new block.

## Publish

The repo now contains a Vercel-ready Next app in `src/app`. The frontend reads
the generated `results/block-N/` files at build time and exposes four routes:

- `/current-block`
- `/previous-blocks`
- `/run-research`
- `/music`

Deploy the Python backend to Railway and set `NEXT_PUBLIC_BACKEND_URL` in Vercel
to the Railway service URL. The backend still writes completed blocks to
`results/block-N/`; commit those generated result files when you want the Vercel
frontend to publish the latest research.

Code and results are published to https://github.com/asimog/cancerhawk
