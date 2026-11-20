# DermaVision AI

A web app that sorts a dermoscopic image into one of the seven HAM10000 lesion classes and shows
the confidence for each. It is a learning and screening-support tool, not a medical device.

- `backend/`  FastAPI service with a CPU PyTorch model (`POST /predict`)
- `frontend/` Next.js site that calls the API

## First-time setup

Install VS Code, Python 3.10 or newer and Node.js 22 LTS or newer. Then, in a terminal in this folder:

    bash setup.sh          # Linux / macOS
    .\setup.ps1            # Windows PowerShell

(If PowerShell blocks the script: `powershell -ExecutionPolicy Bypass -File .\setup.ps1`)

In VS Code, press Ctrl+Shift+P, choose "Python: Select Interpreter" and pick the one in `backend/.venv`.

## Run

VS Code: press Ctrl+Shift+B (task "Run DermaVision (both)"), then open http://localhost:3000.

Or use two terminals:

    cd backend  && .venv/bin/python -m uvicorn app.main:app --reload --env-file .env     # Windows: .venv\Scripts\python.exe
    cd frontend && npm run dev

API documentation: http://localhost:8000/docs

## Free hosting

See `DEPLOY.md` (Vercel for the site, Render for the API, ONNX model).

## Tests

VS Code: Terminal > Run Task > "Backend: run tests" or "Frontend: run tests".

## Settings

- `backend/.env`: `PREPROCESS`, `MODEL_LABELS`, `MODEL_ARCH` (see `.env.example`)
- `frontend/.env.local`: `NEXT_PUBLIC_API_URL`

## Common problems

- "Could not reach the analysis server": open http://localhost:8000/health. If it fails, the API is not running.
- Browser blocks the request: open the site as http://localhost:3000, not 127.0.0.1.
- `venv` fails on Ubuntu or Debian: `sudo apt install python3-venv`, delete `backend/.venv`, run the setup again.
- pip starts downloading gigabytes: stop it and use the setup script, which installs the CPU-only PyTorch.
