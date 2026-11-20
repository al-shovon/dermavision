# DermaVision AI: backend

FastAPI service that classifies one dermoscopic image into the 7 HAM10000 classes.

## Run it locally (CPU is enough)

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate      Linux/macOS:  source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
cp .env.example .env        # Windows: copy .env.example .env
# edit .env: set PREPROCESS and MODEL_LABELS to what the Colab notebook printed
uvicorn app.main:app --reload --port 8000 --env-file .env
```

Open http://localhost:8000/docs and try `POST /predict` with a JPEG or PNG.
The first start downloads the model weights from Hugging Face (about 20 MB) and caches them.

## Settings that depend on your Colab check

`MODEL_LABELS` and `PREPROCESS` must match what the verification notebook printed.
See `.env.example` for all settings.

## Tests

```bash
pytest -q
```

## Swap in your own model later

1. Train in Colab and save the weights with `torch.save(model.state_dict(), "my_model.pth")`.
2. Copy the file to `weights/` and set `MODEL_PATH=weights/my_model.pth`,
   `MODEL_LABELS` (in the order of your training labels) and, after calibration, `TEMPERATURE`.
3. Restart the server. No code changes.

## Production model (ONNX, no PyTorch)

Export `dermavision.onnx` with the last cell of the Colab notebook, put it in `weights/`, and set
`MODEL_PATH=weights/dermavision.onnx` in `.env`. Production installs `requirements-prod.txt` only
(see `Dockerfile`). The deployment steps are in `../DEPLOY.md`.
