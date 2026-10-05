@echo off
REM ==========================================================================
REM  Scan to floor plan: raster wall evidence to clean vector walls + DXF
REM  (c) 2026 Florent Poux - 3D Geodata Academy (learngeodata.eu)
REM
REM  Double-click this file from Explorer. It will:
REM    1. activate the conda "learning_env" environment
REM    2. run walls_to_vectors.py on data\classroom_scene.ply
REM       (writes out\floor_plan.dxf, metrics.json, run_log.txt)
REM    3. replay every stage in an Open3D window: close a window (or press Q)
REM       to move to the next one
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
echo === Step 1/2: from wall pixels to a DXF (walls_to_vectors.py) ===
python walls_to_vectors.py
if errorlevel 1 (
    echo The pipeline failed. See the messages above.
    goto end
)

echo.
echo === Step 2/2: stepped 3D replay, one window per stage ===
python visualize.py

echo.
echo Your drawing: %~dp0out\floor_plan.dxf  (open it in QCAD, LibreCAD, AutoCAD, BricsCAD)

:end
echo.
pause
endlocal
