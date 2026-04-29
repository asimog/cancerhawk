@echo off
setlocal
title CancerHawk - Backend (live API logs)

echo ================================================================
echo   CancerHawk - Autonomous Oncology Research Engine
echo ================================================================
echo.
echo   Backend + UI:  http://localhost:8765
echo   This window streams live API call logs (seq, role, model,
echo   tokens, latency, cost). Press Ctrl+C to stop.
echo ================================================================
echo.

REM --- Sanity check ---
where python >nul 2>&1
if errorlevel 1 (
    echo [X] Python not on PATH. Run install_cancerhawk.bat first.
    pause
    exit /b 1
)

REM --- Free port 8765 if a previous run is still bound to it ---
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8765 ^| findstr LISTENING') do (
    echo [..] Killing stale process %%a on port 8765
    taskkill /F /PID %%a >nul 2>&1
)

REM --- Auto-open browser shortly after the server boots ---
start "" /B cmd /c "timeout /t 3 /nobreak >nul & start http://localhost:8765"

REM --- Run server (foreground; logs visible here) ---
cd /d "%~dp0"
python -m uvicorn app.main:app --host 127.0.0.1 --port 8765 --log-level info --no-access-log

echo.
echo CancerHawk stopped.
pause
