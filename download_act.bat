@echo off
:: One-shot download for the Act inference path.
::
:: Act mode uses the same weights as Cam — the `--allow_act2cam` flag at
:: inference time switches behavior. So this downloads lingbot-world-base-cam
:: (the canonical ckpt_dir per README) plus the separate lingbot-world-base-act
:: repo for setups that want the dedicated Act weights.
::
:: After this script:
::   run_act2cam.sh             (full 8-GPU torchrun, see README)
::   run_act2cam_string.sh      (same, with --action_string user-friendly control)
::
:: Usage:
::   download_act.bat                       default (recommended)
::   download_act.bat C:\my\path            override base-cam local-dir
::   download_act.bat --force               redownload, ignore cache
::   download_act.bat C:\my\path --force    both

setlocal enableextensions enabledelayedexpansion
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8

:: ----- parse args (path + optional --force, in any order) -----
set LOCAL_DIR=
set FORCE_FLAG=
:parse_args
if "%~1"=="" goto args_done
if /I "%~1"=="--force" (
    set FORCE_FLAG=--force
    shift
    goto parse_args
)
set LOCAL_DIR=%~1
shift
goto parse_args
:args_done

:: ----- download base-cam (used as ckpt_dir for Act inference) -----
:: --local-dir applies to base-cam only. base-act lands in the HF cache.
echo === Downloading base-cam (ckpt_dir for Act inference) ===
if defined LOCAL_DIR (
    python download.py --model base-cam --local-dir "!LOCAL_DIR!" !FORCE_FLAG!
) else (
    python download.py --model base-cam !FORCE_FLAG!
)
if errorlevel 1 (
    echo.
    echo ERROR: base-cam download failed.
    exit /b 1
)

:: ----- download dedicated base-act repo (lands in shared HF cache) -----
echo.
echo === Downloading base-act ===
python download.py --model base-act !FORCE_FLAG!
if errorlevel 1 (
    echo.
    echo ERROR: base-act download failed.
    exit /b 1
)

echo.
echo ============================================================
if defined LOCAL_DIR (
    echo base-cam at: !LOCAL_DIR!
) else (
    echo base-cam in HF cache: %USERPROFILE%\.cache\huggingface\hub\models--robbyant--lingbot-world-base-cam
)
echo base-act in HF cache: %USERPROFILE%\.cache\huggingface\hub\models--robbyant--lingbot-world-base-act
echo.
echo Next: run_act2cam.sh (or run_act2cam_string.sh) — see README "LingBot-World-Base (Act)".
echo ============================================================
exit /b 0
