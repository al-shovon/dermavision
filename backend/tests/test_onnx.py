"""Tests for the ONNX path. A tiny hand-made ONNX model stands in for the real one:
a red image scores 'mel', a green image 'nv' and a blue image 'vasc'. That checks the
colour order, the normalisation and the label mapping end to end."""
import io
import os

import numpy as np
import onnx
import pytest
from fastapi.testclient import TestClient
from onnx import TensorProto, helper
from PIL import Image

LABELS = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]


def make_tiny_model(path):
    w = np.zeros((3, 7), dtype=np.float32)
    w[0, 4], w[1, 5], w[2, 6] = 5, 5, 5  # red -> mel, green -> nv, blue -> vasc
    graph = helper.make_graph(
        [
            helper.make_node("GlobalAveragePool", ["input"], ["pooled"]),
            helper.make_node("Flatten", ["pooled"], ["flat"], axis=1),
            helper.make_node("Gemm", ["flat", "W", "b"], ["logits"]),
        ],
        "tiny",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 224, 224])],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, 7])],
        initializer=[
            helper.make_tensor("W", TensorProto.FLOAT, w.shape, w.flatten().tolist()),
            helper.make_tensor("b", TensorProto.FLOAT, [7], [0.0] * 7),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.save(model, str(path))


def png(color):
    buf = io.BytesIO()
    Image.new("RGB", (300, 200), color).save(buf, "PNG")
    return buf.getvalue()


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    path = tmp_path_factory.mktemp("onnx") / "tiny.onnx"
    make_tiny_model(path)
    os.environ["MODEL_PATH"] = str(path)  # the .onnx extension selects the ONNX classifier
    from app.main import app

    with TestClient(app) as c:
        yield c
    del os.environ["MODEL_PATH"]


@pytest.mark.parametrize("color,expected", [((255, 0, 0), "mel"), ((0, 255, 0), "nv"), ((0, 0, 255), "vasc")])
def test_prediction_follows_colour(client, color, expected):
    r = client.post("/predict", files={"file": ("x.png", png(color), "image/png")})
    assert r.status_code == 200
    body = r.json()
    assert body["prediction"]["code"] == expected
    probs = [p["probability"] for p in body["probabilities"]]
    assert abs(sum(probs) - 1) < 0.01 and probs == sorted(probs, reverse=True)
    assert body["uncertain"] is False  # the winning class is far ahead


def test_health_reports_model_name(client):
    assert client.get("/health").json() == {"status": "ok", "model": "tiny.onnx"}


def test_grayscale_and_jpeg_are_accepted(client):
    buf = io.BytesIO()
    Image.new("L", (64, 64), 128).save(buf, "JPEG")
    r = client.post("/predict", files={"file": ("g.jpg", buf.getvalue(), "image/jpeg")})
    assert r.status_code == 200


def test_rejects_wrong_type(client):
    assert client.post("/predict", files={"file": ("n.txt", b"hello", "text/plain")}).status_code == 415


def test_rejects_corrupt_image(client):
    assert client.post("/predict", files={"file": ("x.jpg", b"not an image", "image/jpeg")}).status_code == 400


def test_rejects_large_file(client):
    big = b"\0" * (10 * 1024 * 1024 + 1)
    assert client.post("/predict", files={"file": ("b.png", big, "image/png")}).status_code == 413


def test_missing_model_file_gives_a_clear_error(monkeypatch):
    from app.model_loader import load_classifier

    monkeypatch.setenv("MODEL_PATH", "weights/does-not-exist.onnx")
    with pytest.raises(RuntimeError, match="Model file not found"):
        load_classifier()
