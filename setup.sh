#!/usr/bin/env bash
# One-time setup for Linux and macOS. Run it from this folder:  bash setup.sh
set -euo pipefail
cd "$(dirname "$0")"

echo "== Backend: virtual environment and packages =="
cd backend
python3 -m venv .venv
PY=".venv/bin/python"
"$PY" -m pip install --upgrade pip
if [ "$(uname -s)" = "Linux" ]; then
  # CPU-only PyTorch. Without this index pip downloads gigabytes of CUDA libraries.
  "$PY" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
else
  "$PY" -m pip install torch torchvision
fi
"$PY" -m pip install -r requirements.txt
[ -f .env ] || cp .env.example .env
cd ..

echo "== Frontend: npm packages =="
cd frontend
npm install
[ -f .env.local ] || cp .env.local.example .env.local
cd ..

echo
echo "Setup finished."
echo "In VS Code: select the Python interpreter (backend/.venv), then press Ctrl+Shift+B to start the API and the site."
