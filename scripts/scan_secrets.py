#!/usr/bin/env python3
"""Scan for secrets accidentally committed to tracked files."""
import os
import sys
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Patterns that look like secrets (prefix + high-entropy suffix)
SECRET_PATTERNS = [
    (r'sk-or-v1-[a-zA-Z0-9]{32,}', "OpenRouter API key"),
    (r'ghp_[a-zA-Z0-9]{36}', "GitHub personal access token"),
    (r'github_pat_[a-zA-Z0-9_]{22,}', "GitHub fine-grained token"),
    (r'moltbook_sk_[a-zA-Z0-9]{16,}', "Moltbook API key"),
    (r'sol_pri_[a-zA-Z0-9]{32,}', "Solana private key (hex)"),
    (r'[1-9A-HJ-NP-Za-km-z]{87,88}', "Solana base58 private key (long)"),
]

IGNORE_DIRS = {'.git', '.next', 'node_modules', '__pycache__', '.pytest_cache',
               '.vercel', 'results', 'public/music'}
IGNORE_FILES = {'.env', '.env.local', '.env.example', 'keyfile.json',
                '.cancerhawk-wallet.json', '.sub-agent-wallet-1.json',
                '.sub-agent-wallet-2.json', '.sub-agent-wallet-3.json'}

EXIT_CODE = 0

for dirpath, dirnames, filenames in os.walk(ROOT):
    dirnames[:] = [d for d in dirnames if d not in IGNORE_DIRS]
    for fname in filenames:
        if fname in IGNORE_FILES:
            continue
        fpath = Path(dirpath) / fname
        if fpath.suffix not in {'.py', '.js', '.ts', '.tsx', '.json', '.yml', '.yaml',
                                '.toml', '.sh', '.md', '.txt', '.env', '.cfg', '.ini',
                                '.mjs', '.Dockerfile'}:
            if fname not in ('Dockerfile', 'Procfile', '.gitignore', '.railwayignore'):
                continue
        try:
            content = fpath.read_text(encoding='utf-8', errors='replace')
        except Exception:
            continue
        for pattern, label in SECRET_PATTERNS:
            for match in re.finditer(pattern, content):
                # Check if it's inside a comment or test fixture
                line_start = content.rfind('\n', 0, match.start()) + 1
                line = content[line_start:content.find('\n', match.end())]
                if any(skip in line.lower() for skip in ('test', 'example', 'placeholder', '...', 'mock')):
                    continue
                rel = fpath.relative_to(ROOT)
                print(f"WARNING: possible {label} in {rel}:{line[:120]}")
                EXIT_CODE = 1

if EXIT_CODE:
    print("\nSecrets scan FAILED — review the warnings above.")
else:
    print("Secrets scan PASSED — no exposed secrets found.")

sys.exit(EXIT_CODE)
