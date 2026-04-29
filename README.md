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

## Publish

Pushes to `master` deploy the `results/` directory to GitHub Pages. That means
the public site root serves the latest generated CancerHawk block instead of the
developer control panel or old repository dashboard.

Code and results are published to https://github.com/asimog/cancerhawk
