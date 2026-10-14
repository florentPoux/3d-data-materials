@echo off
REM ==========================================================================
REM  Two faces, one wall: scan to IFC wall thickness - one-click reproduction
REM  (c) 2026 Florent Poux - 3D Geodata Academy (learngeodata.eu)
REM
REM  Double-click this file from Explorer. It will:
REM    1. activate the conda "learning_env" environment
REM    2. run the pipeline on data\s3dis_office.ply (about one second)
REM       and write out\measured_partitions.ifc
REM    3. replay each stage in its own 3D window: close a window to advance
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
echo === Step 1/2: measuring the walls and writing the IFC (tests: python -m pytest tests -q) ===
python wall_thickness_to_ifc.py
if errorlevel 1 (
    echo The pipeline failed. See the messages above.
    goto end
)

echo.
echo === Step 2/2: stepped 3D replay (close each window to advance) ===
python visualize.py

:end
echo.
pause
endlocal
