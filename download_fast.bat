@echo off
:: One-shot setup for the LingBot-World-Fast inference path on Windows.
::
:: 1. Downloads the fast model (~73 GB) into the shared HF cache.
:: 2. Builds .\fast-mini-cam\ — a composite ckpt dir that nests the fast
::    snapshot AND hardlinks/junctions T5 + VAE + tokenizer from local donor
::    copies (no re-download). The literal "cam" in the dir name is required:
::    wan/image2video_fast.py line 95 sniffs the path to set control_type.
::
:: After this script, run:   test_fast.bat
::
:: Usage:
::   download_fast.bat                       default (recommended)
::   download_fast.bat C:\my\path            override fast model local-dir
::
:: All hardlinks/junctions are on the same NTFS volume, no admin required.

setlocal enableextensions enabledelayedexpansion
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8

:: ----- 1. download fast model -----
set LOCAL_DIR=%~1
if defined LOCAL_DIR (
    python download.py --model fast --local-dir "%LOCAL_DIR%"
) else (
    python download.py --model fast
)
if errorlevel 1 (
    echo.
    echo ERROR: fast model download failed.
    exit /b 1
)

:: ----- 2. build fast-mini-cam -----
:: Every required aux file (T5, VAE, tokenizer) is already inside the fast
:: snapshot, so link from there directly — no donor lookups, no re-download.
set DST=%~dp0fast-mini-cam
set FAST_CACHE=%USERPROFILE%\.cache\huggingface\hub\models--robbyant--lingbot-world-fast\snapshots

if not exist "%DST%" mkdir "%DST%"
if not exist "%DST%\google" mkdir "%DST%\google"

:: locate the fast snapshot
set SNAP=
for /f "delims=" %%S in ('dir /b /ad "%FAST_CACHE%" 2^>nul') do set SNAP=%FAST_CACHE%\%%S
if not defined SNAP (
    echo ERROR: no snapshot inside %FAST_CACHE%
    exit /b 2
)

:: T5 (hardlink from snapshot — same NTFS volume)
if not exist "%DST%\models_t5_umt5-xxl-enc-bf16.pth" (
    mklink /H "%DST%\models_t5_umt5-xxl-enc-bf16.pth" "%SNAP%\models_t5_umt5-xxl-enc-bf16.pth" >nul || ( echo ERROR linking T5 & exit /b 1 )
    echo Linked T5 from snapshot.
) else (
    echo T5 already present.
)

:: VAE (hardlink from snapshot)
if not exist "%DST%\Wan2.1_VAE.pth" (
    mklink /H "%DST%\Wan2.1_VAE.pth" "%SNAP%\Wan2.1_VAE.pth" >nul || ( echo ERROR linking VAE & exit /b 1 )
    echo Linked VAE from snapshot.
) else (
    echo VAE already present.
)

:: tokenizer (junction from snapshot)
if not exist "%DST%\google\umt5-xxl\" (
    mklink /J "%DST%\google\umt5-xxl" "%SNAP%\google\umt5-xxl" >nul || ( echo ERROR linking tokenizer & exit /b 1 )
    echo Junctioned tokenizer from snapshot.
) else (
    echo Tokenizer already present.
)

:: fast model dir (junction into the snapshot — the 16 safetensors live there)
if exist "%DST%\lingbot_world_fast" rmdir "%DST%\lingbot_world_fast" 2>nul
mklink /J "%DST%\lingbot_world_fast" "%SNAP%" >nul || ( echo ERROR linking fast snapshot & exit /b 1 )
echo Junctioned lingbot_world_fast -^> %SNAP%

echo.
echo ============================================================
echo Ready. ckpt_dir = %DST%
echo Run:    test_fast.bat
echo ============================================================
exit /b 0
