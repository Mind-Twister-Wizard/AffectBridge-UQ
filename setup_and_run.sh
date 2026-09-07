#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PYTHON_BIN="${PYTHON_BIN:-python3}"
if [[ ! -x ".venv/bin/python" ]]; then
  "$PYTHON_BIN" -m venv .venv
fi
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
if command -v nvidia-smi >/dev/null 2>&1; then
  if ! .venv/bin/python -c "import torch; raise SystemExit(0 if torch.__version__.startswith('2.6.0') and torch.cuda.is_available() else 1)" >/dev/null 2>&1; then
    .venv/bin/python -m pip install --upgrade --force-reinstall torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
  fi
else
  .venv/bin/python -m pip install --upgrade torch==2.6.0 torchaudio==2.6.0
fi
.venv/bin/python -c "import torch; print('PyTorch', torch.__version__, '| CUDA available=', torch.cuda.is_available()); print('Device=', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
.venv/bin/python run_all.py --download --train --report --ablation-scope representative
echo "Finished. Open outputs/final_report.html"
