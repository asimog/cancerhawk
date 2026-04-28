@echo off
setlocal

cls
echo ================================================================
echo   CancerHawk LAUNCHER
echo ================================================================
echo.

REM ================================================================
REM STEP 1: Check Python Installation
REM ================================================================
echo [1/5] Checking Python installation...
where python >nul 2>&1
if errorlevel 1 (
    echo.
    echo ============================================================
    echo ERROR: Python is not installed or not in PATH
    echo ============================================================
    echo.
    echo Please install Python 3.8+ from: https://www.python.org/downloads/
    echo IMPORTANT: Check 'Add Python to PATH' during installation
    echo.
    pause
    exit /b 1
)
python --version
echo Python found!
echo.

REM ================================================================
REM STEP 2: Create Necessary Directories
REM ================================================================
echo [2/5] Creating necessary directories...
if not exist "results" mkdir "results"
echo Directories created successfully!
echo.

REM ================================================================
REM STEP 3: Install Python Dependencies
REM ================================================================
echo [3/5] Installing Python dependencies...
echo This may take a few minutes if this is your first time...
echo.
python -m pip install --upgrade pip >nul 2>&1
pip install --upgrade -r app\requirements.txt
if errorlevel 1 (
    echo.
    echo ============================================================
    echo ERROR: Failed to install Python dependencies
    echo ============================================================
    echo.
    echo Please check:
    echo - Internet connection is working
    echo - You have permission to install packages
    echo - requirements.txt exists in the current directory
    echo.
    pause
    exit /b 1
)
echo Python dependencies installed successfully!
echo.

REM ================================================================
REM STEP 4: Clean Up Existing Process
REM ================================================================
echo [4/5] Cleaning up existing process on port 8765...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8765 ^| findstr LISTENING') do (
    echo Found process %%a using port 8765, terminating...
    taskkill /F /PID %%a >nul 2>&1
)
timeout /t 2 /nobreak >nul
echo Port 8765 cleaned successfully!
echo.

REM ================================================================
REM STEP 5: Start Services
REM ================================================================
echo [5/5] Starting CancerHawk...
echo.
echo ================================================================
echo   CancerHawk STARTING
echo ================================================================
echo.
echo Backend + Frontend will run on: http://localhost:8765
echo.
echo A window will open titled "CancerHawk Backend".
echo This window shows real-time API call logs including:
echo   - seq, role, model
echo   - prompt tokens / completion tokens
echo   - latency (ms), cost (USD)
echo   - status (ok/fail)
echo.
echo The browser will open automatically to the UI.
echo.
echo Starting in 3 seconds... (Ctrl+C to cancel)
timeout /t 3 /nobreak >nul
echo.

REM Start backend in separate window; this serves both API and static frontend
echo Starting CancerHawk backend...
start "CancerHawk Backend" cmd /k "cd /d "%~dp0" && python -m uvicorn app.main:app --host 127.0.0.1 --port 8765 --log-level info --no-access-log"

REM Wait for backend to initialize
echo Waiting for backend to initialize...
timeout /t 5 /nobreak >nul

REM Open browser
echo Opening browser to http://localhost:8765 ...
start http://localhost:8765

echo.
echo ================================================================
echo   CancerHawk STARTED!
echo ================================================================
echo.
echo Two windows should now be open:
echo   1) "CancerHawk Backend" — shows live API call logs
echo   2) Your web browser — the CancerHawk UI
echo.
echo To stop: Close the "CancerHawk Backend" window (or press Ctrl+C in it)
echo.
echo This launcher window can now be closed.
echo.
echo Closing launcher window automatically in 3 seconds...
timeout /t 3 /nobreak >nul
exit /b 0
