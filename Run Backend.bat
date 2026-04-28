@echo off
setlocal

title CancerHawk Backend — API Call Logs

echo ================================================================
echo   CancerHawk Backend — Live API Call Stream
echo ================================================================
echo.
echo This window displays real-time logs from the CancerHawk backend.
echo Each API call shows: sequence, role, model, tokens in/out,
echo latency, cost, and status.
echo.
echo Press Ctrl+C to stop the server.
echo ================================================================
echo.

REM Change to project root directory
cd /d "%~dp0"

REM Start backend (serves both API and static frontend)
python -m uvicorn app.main:app --host 127.0.0.1 --port 8765 --log-level info --no-access-log

pause
