@echo off
rem P24 article 04: how far to the way out. Double-click to run the pipeline, then the 3D replay.
call "%USERPROFILE%\miniconda3\Scripts\activate.bat" learning_env
cd /d %~dp0
if not exist data\s3dis_area1_open_5cm.ply (
    echo The Area 1 floor is missing. Request S3DIS ^(see README.md^), then run:
    echo     python prepare_s3dis.py --s3dis path\to\Stanford3dDataset_v1.2_Aligned_Version
    pause
    exit /b 1
)
python evacuation_distance.py
if errorlevel 1 (
    echo The pipeline stopped with an error. Read the message above.
    pause
    exit /b 1
)
echo.
echo Numbers: out\metrics.json. Now the 3D replay: close each window to go to the next step.
python visualize.py
pause
