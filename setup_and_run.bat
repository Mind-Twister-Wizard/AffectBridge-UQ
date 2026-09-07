@echo off
setlocal EnableExtensions
cd /d "%~dp0"

echo ===============================================================
echo AffectBridge-UQ v2 - paper-quality one-click experiment
echo ===============================================================

where py >nul 2>nul
if errorlevel 1 (
  set "PY_LAUNCH=python"
) else (
  py -3.10 -c "import sys" >nul 2>nul
  if errorlevel 1 (set "PY_LAUNCH=py") else (set "PY_LAUNCH=py -3.10")
)

if not exist ".venv\Scripts\python.exe" (
  echo Creating virtual environment...
  %PY_LAUNCH% -m venv .venv
  if errorlevel 1 goto :error
)

echo Installing/updating core dependencies...
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 goto :error

where nvidia-smi >nul 2>nul
if not errorlevel 1 (
  echo NVIDIA GPU detected. Ensuring PyTorch 2.6 CUDA 12.4 build...
  .venv\Scripts\python.exe -c "import torch,sys; sys.exit(0 if torch.__version__.startswith('2.6.0') and torch.cuda.is_available() else 1)" >nul 2>nul
  if errorlevel 1 (
    .venv\Scripts\python.exe -m pip install --upgrade --force-reinstall torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
    if errorlevel 1 goto :error
  )
) else (
  echo No NVIDIA driver detected. Installing CPU PyTorch 2.6...
  .venv\Scripts\python.exe -m pip install --upgrade torch==2.6.0 torchaudio==2.6.0
  if errorlevel 1 goto :error
)

.venv\Scripts\python.exe -c "import torch; print('PyTorch', torch.__version__, '| CUDA available=', torch.cuda.is_available()); print('Device=', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
if errorlevel 1 goto :error

echo.
echo Running final paper pipeline. Completed fold/seed runs are resumed instead of repeated.
echo First v2 run will create the new multi-layer WavLM cache once.
.venv\Scripts\python.exe run_all.py --download --train --report --ablation-scope representative
if errorlevel 1 goto :error

echo.
echo ===============================================================
echo Finished. Open outputs\final_report.html
echo Main paper table: outputs\paper_main_results.csv
echo ===============================================================
pause
exit /b 0

:error
echo.
echo The pipeline stopped. Existing completed runs are preserved and will be resumed next time.
echo Read the error above; common causes are Kaggle credentials or Python installation.
pause
exit /b 1
