@echo off
:: Smallest-possible smoke test for the lingbot-world "fast" inference path.
:: One GPU, smallest size, smallest 4n+1 frame count. ~minutes, not hours.
::
:: Usage:
::   test_fast.bat              run with defaults (examples/03, 21 frames, 480*832)
::   test_fast.bat 41           override frame_num (must be 4n+1)
::   test_fast.bat 21 04        override frame_num + example folder (00..05)

setlocal enableextensions
cd /d "%~dp0"

set FRAMES=%~1
set EX=%~2
set CKPT=%~3
if not defined FRAMES set FRAMES=21
if not defined EX     set EX=03

:: Resolve checkpoint dir. Priority:
::   1. 3rd positional arg
::   2. LINGBOT_FAST_CKPT env var
::   3. .\fast\ next to this script (flat layout)
::   4. HF cache snapshot at ~/.cache/huggingface/hub/models--robbyant--lingbot-world-fast/snapshots/<rev>/
if not defined CKPT if defined LINGBOT_FAST_CKPT set CKPT=%LINGBOT_FAST_CKPT%
if not defined CKPT if exist "%~dp0fast-mini-cam\lingbot_world_fast\" set CKPT=%~dp0fast-mini-cam
if not defined CKPT if exist "%~dp0lingbot-world-base-cam\lingbot_world_fast\" set CKPT=%~dp0lingbot-world-base-cam
if not defined CKPT if exist "%~dp0fast\" set CKPT=%~dp0fast
if not defined CKPT (
    set HF_FAST_ROOT=%USERPROFILE%\.cache\huggingface\hub\models--robbyant--lingbot-world-fast\snapshots
    if exist "%USERPROFILE%\.cache\huggingface\hub\models--robbyant--lingbot-world-fast\snapshots\" (
        for /f "delims=" %%S in ('dir /b /ad "%USERPROFILE%\.cache\huggingface\hub\models--robbyant--lingbot-world-fast\snapshots"') do (
            set CKPT=%USERPROFILE%\.cache\huggingface\hub\models--robbyant--lingbot-world-fast\snapshots\%%S
        )
    )
)

set EX_DIR=%~dp0examples\%EX%
set OUT=%~dp0output_test
set LOG=%~dp0test_fast.log
set PY=python

if not defined CKPT (
    echo ERROR: no ckpt dir found.
    echo Tried: 3rd arg, LINGBOT_FAST_CKPT env, .\fast\, HF cache.
    echo Download it first: python download.py --model fast
    exit /b 2
)
if not exist "%CKPT%\" (
    echo ERROR: ckpt dir does not exist: %CKPT%
    exit /b 2
)

:: generate_fast.py expects ckpt_dir to be the BASE-CAM weights dir, with the
:: fast model nested at <ckpt_dir>\lingbot_world_fast\. The standalone fast
:: repo lacks T5/VAE and will FileNotFoundError on models_t5_umt5-xxl-enc-bf16.pth.
if not exist "%CKPT%\models_t5_umt5-xxl-enc-bf16.pth" (
    echo ERROR: %CKPT% is missing T5 weights ^(models_t5_umt5-xxl-enc-bf16.pth^).
    echo You probably pointed at the fast-only snapshot. The fast model nests
    echo inside lingbot-world-base-cam; download base-cam first:
    echo.
    echo   huggingface-cli download robbyant/lingbot-world-base-cam --local-dir .\lingbot-world-base-cam
    echo   huggingface-cli download robbyant/lingbot-world-fast      --local-dir .\lingbot-world-base-cam\lingbot_world_fast
    echo.
    echo Then: test_fast.bat 21 03 .\lingbot-world-base-cam
    exit /b 2
)
if not exist "%EX_DIR%\image.jpg" (
    echo ERROR: example folder missing image.jpg: %EX_DIR%
    exit /b 2
)
if not exist "%OUT%\" mkdir "%OUT%"

echo ============================================================
echo lingbot-world fast smoke test
echo ============================================================
echo   ckpt      : %CKPT%
echo   example   : %EX_DIR%
echo   frame_num : %FRAMES%   (must be 4n+1)
echo   size      : 480*832
echo   out_dir   : %OUT%
echo   log       : %LOG%
echo ============================================================

%PY% generate_fast.py --task i2v-A14B --size 480*832 --ckpt_dir "%CKPT%" --image "%EX_DIR%\image.jpg" --action_path "%EX_DIR%" --frame_num %FRAMES% --save_dir "%OUT%" --base_seed 42 --prompt "A smoke-test clip; minimal frames, smallest resolution; ignore content quality." 1>"%LOG%" 2>&1

set EC=%ERRORLEVEL%
echo.
echo ============================================================
if %EC%==0 (
    echo OK. Result video^(s^) in %OUT%\
    dir /b "%OUT%" 2>nul
) else (
    echo FAIL ^(exit %EC%^). Last log lines:
    powershell -NoProfile -Command "Get-Content -LiteralPath '%LOG%' -Tail 30"
)
echo ============================================================
exit /b %EC%
