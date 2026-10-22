@echo off
REM ==========================================================================
REM  Accessibility audit from a scan - one-click reproduction
REM  (c) 2026 Florent Poux - 3D Geodata Academy (learngeodata.eu)
REM
REM  Double-click this file from Explorer. It will:
REM    1. activate the conda "learning_env" environment
REM    2. run the audit on data\s3dis_area2_access.ply (about 3 seconds)
REM    3. replay it step by step: one 3D window per step (scan, grid, clear
REM       width, centre space, doors, turning floor, reach, numbered issues).
REM       Close a window (or press Q) to advance to the next.
REM ==========================================================================
setlocal
cd /d "%~dp0"

echo.
echo === Activating conda environment: learning_env ===
call "%USERPROFILE%\miniconda3\Scripts\activate.bat" learning_env
if errorlevel 1 (
    echo Could not activate learning_env. Check your Miniconda path.
    goto end
)

echo.
echo === Step 1/2: running the audit (accessibility_audit.py) ===
python accessibility_audit.py
if errorlevel 1 (
    echo The audit failed. See the messages above.
    goto end
)

echo.
echo === Step 2/2: guided 3D replay, one window per step ===
python visualize.py

echo.
echo The numbered issue list is in out\issues.csv
:end
echo.
pause
endlocal
