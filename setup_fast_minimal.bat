@echo off
:: Build a minimal ckpt_dir for generate_fast.py that AVOIDS downloading the
:: full lingbot-world-base-cam repo (~50 GB). We need only:
::   - T5 encoder + tokenizer  (~5-6 GB)
::   - Wan2.1 VAE              (~250 MB)
::   - the already-downloaded fast snapshot (~73 GB, reused via symlink)
::
:: Result layout:
::   .\fast-mini-cam\
::      models_t5_umt5-xxl-enc-bf16.pth
::      Wan2.1_VAE.pth
::      google\umt5-xxl\<tokenizer files>
::      lingbot_world_fast\  <-- junction into existing HF cache snapshot
::
:: 'cam' must appear in the dir name (wan/image2video_fast.py line 95 sniffs
:: the path to choose camera-pose mode vs act mode).

setlocal enableextensions enabledelayedexpansion
cd /d "%~dp0"

set DST=%~dp0fast-mini-cam
set FAST_CACHE=%USERPROFILE%\.cache\huggingface\hub\models--robbyant--lingbot-world-fast\snapshots

if not exist "%DST%" mkdir "%DST%"

:: 1. Pull only the auxiliary files from base-cam. --include filters at the
::    huggingface-cli level so we don't pay for the 14B noise models.
echo Downloading T5 from robbyant/lingbot-world-base-cam...
hf download robbyant/lingbot-world-base-cam --include "models_t5_umt5-xxl-enc-bf16.pth" --local-dir "%DST%"
if errorlevel 1 ( echo ERROR: T5 download failed. & exit /b 1 )

echo Downloading VAE...
hf download robbyant/lingbot-world-base-cam --include "Wan2.1_VAE.pth" --local-dir "%DST%"
if errorlevel 1 ( echo ERROR: VAE download failed. & exit /b 1 )

echo Downloading T5 tokenizer...
hf download robbyant/lingbot-world-base-cam --include "google/umt5-xxl/*" --local-dir "%DST%"
if errorlevel 1 (
    echo ERROR: hf-cli download failed.
    exit /b 1
)

:: 2. Link the existing fast snapshot in as a subdir.
if not exist "%FAST_CACHE%\" (
    echo ERROR: fast snapshot not in HF cache: %FAST_CACHE%
    echo Run download_fast.bat first.
    exit /b 2
)
set SNAP=
for /f "delims=" %%S in ('dir /b /ad "%FAST_CACHE%"') do set SNAP=%FAST_CACHE%\%%S
if not defined SNAP (
    echo ERROR: no snapshot inside %FAST_CACHE%
    exit /b 2
)

if exist "%DST%\lingbot_world_fast" rmdir "%DST%\lingbot_world_fast"
mklink /J "%DST%\lingbot_world_fast" "%SNAP%"
if errorlevel 1 (
    echo ERROR: mklink failed. Run this script from an elevated cmd, or copy manually:
    echo   xcopy /E /I /Y "%SNAP%" "%DST%\lingbot_world_fast"
    exit /b 1
)

echo.
echo ============================================================
echo Minimal ckpt dir ready at: %DST%
echo Run with:
echo   test_fast.bat 21 03 "%DST%"
echo ============================================================
exit /b 0
