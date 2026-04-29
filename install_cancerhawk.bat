@echo off
setlocal
title CancerHawk - Install

echo ================================================================
echo   CancerHawk - One-Time Installer
echo ================================================================
echo.

REM --- Check Python ---
where python >nul 2>&1
if errorlevel 1 (
    echo [X] Python is not installed or not on PATH.
    echo     Install Python 3.10+ from https://www.python.org/downloads/
    echo     IMPORTANT: tick "Add Python to PATH" during install.
    pause
    exit /b 1
)
for /f "tokens=*" %%v in ('python --version 2^>^&1') do echo [OK] %%v

REM --- Upgrade pip + install requirements ---
echo.
echo [..] Upgrading pip...
python -m pip install --upgrade pip
if errorlevel 1 goto :pip_fail

echo.
echo [..] Installing CancerHawk dependencies (fastapi, uvicorn, httpx, ...)
python -m pip install --upgrade -r "%~dp0app\requirements.txt"
if errorlevel 1 goto :pip_fail

REM --- Create runtime dirs ---
if not exist "%~dp0results" mkdir "%~dp0results"

echo.
echo ================================================================
echo   Install complete. Run "run_cancerhawk.bat" to start the engine.
echo ================================================================
pause
exit /b 0

:pip_fail
echo.
echo [X] pip install failed. Check your internet connection / permissions.
pause
exit /b 1
