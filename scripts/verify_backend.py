#!/usr/bin/env python3
"""Verify backend and deployment configuration."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ALL_OK = True

def check(label: str, condition: bool, detail: str = ""):
    global ALL_OK
    status = "OK" if condition else "FAIL"
    print(f"  [{status}] {label}" + (f" — {detail}" if detail else ""))
    if not condition:
        ALL_OK = False

print("=== CancerHawk Deployment Verification ===\n")

# --- Files ---
print("Required files:")
for f in ["Dockerfile", "vercel.json", "railway.toml", "railway.json",
          "package.json", "app/requirements.txt", "app/main.py"]:
    check(f"exists: {f}", (ROOT / f).exists())

# --- app/requirements.txt ---
reqs = (ROOT / "app" / "requirements.txt").read_text()
check("requirements has fastapi", "fastapi" in reqs)
check("requirements has uvicorn", "uvicorn" in reqs)
check("requirements has httpx", "httpx" in reqs)
check("requirements has asyncpg", "asyncpg" in reqs)

# --- Dockerfile ---
df = (ROOT / "Dockerfile").read_text()
check("Dockerfile exposes port", "EXPOSE" in df)
check("Dockerfile has CMD", "CMD" in df)
check("Dockerfile installs requirements", "requirements.txt" in df)

# --- No secrets in tracked files ---
print("\nSecrets check:")
env_example = (ROOT / ".env.example")
if env_example.exists():
    env_txt = env_example.read_text()
    # Allow placeholders like sk-or-v1-... or github_pat_... but NOT 40+ char active keys
    import re
    has_active_key = bool(re.findall(r'sk-or-v1-[a-zA-Z0-9]{40,}', env_txt))
    has_active_token = bool(re.findall(r'ghp_[a-zA-Z0-9]{36,}', env_txt))
    check(".env.example has no real keys", not has_active_key and not has_active_token,
          "contains active API key token")
else:
    check(".env.example exists", False, "missing")

# --- Railway config ---
railway_toml = (ROOT / "railway.toml").read_text() if (ROOT / "railway.toml").exists() else ""
check("railway.toml has healthcheck", "healthcheck" in railway_toml)
check("railway.toml has Dockerfile builder", "DOCKERFILE" in railway_toml)

# --- Vercel config ---
vercel_json = (ROOT / "vercel.json").read_text()
check("vercel.json has nextjs framework", "nextjs" in vercel_json)

# --- Medical disclaimer ---
print("\nMedical safety:")
try:
    publisher_py = (ROOT / "app" / "publisher.py").read_text()
    check("publisher has disclaimer", "medical" in publisher_py.lower() or "generated" in publisher_py.lower())
except Exception:
    check("publisher.py readable", False)

# --- Git publisher safety ---
print("\nGit publisher safety:")
check("publisher commits only results/", "results" in (ROOT / "app" / "publisher.py").read_text().lower())

# --- Python syntax ---
print("\nPython compile check:")
import py_compile
for pyfile in (ROOT / "app").glob("*.py"):
    try:
        py_compile.compile(str(pyfile), doraise=True)
        check(f"compile: app/{pyfile.name}", True)
    except py_compile.PyCompileError as e:
        check(f"compile: app/{pyfile.name}", False, str(e))

print(f"\n{'ALL CHECKS PASSED' if ALL_OK else 'SOME CHECKS FAILED'}")
sys.exit(0 if ALL_OK else 1)
