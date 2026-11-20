import io
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError

from .classes import CLASS_INFO, DISCLAIMER
from .model_loader import load_classifier

MAX_BYTES = 10 * 1024 * 1024  # FR-02: 10 MB upload limit
MAX_PIXELS = 25_000_000
UNCERTAIN_BELOW = float(os.getenv("UNCERTAIN_BELOW", "0.60"))  # FR-08


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_classifier()  # load once at start-up and keep it in memory
    yield


app = FastAPI(title="DermaVision AI", version="0.1.0", lifespan=lifespan)


def get_classifier():
    # loaded on first use if the host does not run start-up events (some serverless hosts)
    if getattr(app.state, "clf", None) is None:
        app.state.clf = load_classifier()
    return app.state.clf

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",")],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok", "model": get_classifier().name}


@app.post("/predict")
def predict(file: UploadFile = File(...)):
    if file.content_type not in ("image/jpeg", "image/png"):
        raise HTTPException(415, "Please upload a JPEG or PNG image.")
    data = file.file.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "The file is larger than 10 MB.")
    try:
        img = Image.open(io.BytesIO(data))
        if img.width * img.height > MAX_PIXELS:
            raise HTTPException(413, "The image has too many pixels.")
        img = img.convert("RGB")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
        raise HTTPException(400, "The file could not be read as an image.")

    clf = get_classifier()
    probs = clf.predict(img)  # nothing is written to disk
    ranked = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
    code, confidence = ranked[0]
    return {
        "model": clf.name,
        "prediction": {"code": code, **CLASS_INFO[code], "confidence": round(confidence, 4)},
        "uncertain": confidence < UNCERTAIN_BELOW,
        "probabilities": [
            {"code": c, "name": CLASS_INFO[c]["name"], "probability": round(p, 4)} for c, p in ranked
        ],
        "disclaimer": DISCLAIMER,
    }
