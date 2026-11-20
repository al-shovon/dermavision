"""API tests. They use a randomly initialised EfficientNet-B0, so no download is needed."""
import io
import os

import pytest
import timm
import torch
from fastapi.testclient import TestClient
from PIL import Image


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    weights = tmp_path_factory.mktemp("weights") / "random.pth"
    torch.save(timm.create_model("efficientnet_b0", pretrained=False, num_classes=7).state_dict(), weights)
    os.environ["MODEL_PATH"] = str(weights)
    from app.main import app

    with TestClient(app) as c:  # runs the lifespan, which loads the model
        yield c
    del os.environ["MODEL_PATH"]


def image_bytes(fmt="PNG", size=(320, 240)):
    buf = io.BytesIO()
    Image.new("RGB", size, (170, 110, 90)).save(buf, fmt)
    return buf.getvalue()


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


@pytest.mark.parametrize("fmt,mime", [("PNG", "image/png"), ("JPEG", "image/jpeg")])
def test_predict_ok(client, fmt, mime):
    r = client.post("/predict", files={"file": ("lesion", image_bytes(fmt), mime)})
    assert r.status_code == 200
    body = r.json()
    probs = [p["probability"] for p in body["probabilities"]]
    assert len(probs) == 7
    assert abs(sum(probs) - 1) < 0.01
    assert probs == sorted(probs, reverse=True)
    assert body["prediction"]["confidence"] == probs[0]
    assert "disclaimer" in body


def test_rejects_wrong_type(client):
    r = client.post("/predict", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert r.status_code == 415


def test_rejects_corrupt_image(client):
    r = client.post("/predict", files={"file": ("x.jpg", b"not an image", "image/jpeg")})
    assert r.status_code == 400


def test_rejects_large_file(client):
    big = b"\0" * (10 * 1024 * 1024 + 1)
    r = client.post("/predict", files={"file": ("big.png", big, "image/png")})
    assert r.status_code == 413
