@echo off
setlocal enabledelayedexpansion
title CancerHawk Engine

echo.
echo ========================================================
echo   CancerHawk - Autonomous Oncology Research Engine
echo ========================================================
echo.

REM --- Check Python ---
where python >nul 2>nul
if errorlevel 1 (
    echo [X] Python is not installed or not on PATH.
    echo     Install Python 3.10+ from https://www.python.org/downloads/
    echo     and re-run this script.
    pause
    exit /b 1
)

for /f "tokens=2" %%v in ('python --version 2^>^&1') do set PYVER=%%v
echo [OK] Python detected: !PYVER!
echo.

REM --- Install dependencies ---
echo [..] Installing dependencies (fastapi, uvicorn, httpx)...
python -m pip install --upgrade pip --quiet
python -m pip install -r "%~dp0app\requirements.txt"
if errorlevel 1 (
    echo.
    echo [X] Dependency install failed. See errors above.
    pause
    exit /b 1
)
echo [OK] Dependencies ready.
echo.

REM --- Launch ---
echo ========================================================
echo   Starting CancerHawk on http://localhost:8765
echo ========================================================
echo.
echo   Open this URL in your browser:
echo.
echo     http://localhost:8765
echo.
echo   Paste your OpenRouter API key (sk-or-v1-...) in the UI,
echo   pick a research goal, click Run.
echo.
echo   Press Ctrl+C in this window to stop the server.
echo ========================================================
echo.

REM Try to auto-open the browser after a short delay
start "" /B cmd /c "timeout /t 3 /nobreak >nul & start http://localhost:8765"

python "%~dp0app\main.py"

echo.
echo CancerHawk stopped.
pause
