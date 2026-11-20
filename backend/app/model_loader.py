"""Builds the classifier from environment variables.

MODEL_FORMAT=onnx      ONNX Runtime, no PyTorch needed (production). Chosen automatically for .onnx files.
MODEL_FORMAT=pytorch   PyTorch checkpoint (local development, default).
"""
import os
from pathlib import Path


def load_classifier():
    path = os.getenv("MODEL_PATH")
    fmt = (os.getenv("MODEL_FORMAT") or ("onnx" if path and path.lower().endswith(".onnx") else "pytorch")).lower()
    labels = [x.strip() for x in os.getenv("MODEL_LABELS", "akiec,bcc,bkl,df,mel,nv,vasc").split(",")]
    preprocess = os.getenv("PREPROCESS", "squash")
    size = int(os.getenv("IMAGE_SIZE", "224"))
    temperature = float(os.getenv("TEMPERATURE", "1.0"))

    if fmt == "onnx":
        if not path:
            raise RuntimeError("MODEL_FORMAT=onnx needs MODEL_PATH, for example weights/dermavision.onnx")
        if not Path(path).is_file():
            raise RuntimeError(f"Model file not found: {path}. Export it from Colab and put it in backend/weights/.")
        from .onnx_model import OnnxClassifier

        return OnnxClassifier(path, labels, preprocess, size, temperature, name=os.getenv("MODEL_NAME", Path(path).name))

    from .skin_model import SkinClassifier

    repo = os.getenv("MODEL_REPO", "pthennakoon25/skinvision-ai-model")
    if not path:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(repo, os.getenv("MODEL_FILE", "best_model.pth"))
        default_name = repo
    else:
        default_name = Path(path).name
    return SkinClassifier(
        weights_path=path,
        labels=labels,
        arch=os.getenv("MODEL_ARCH", "efficientnet_b0"),
        preprocess=preprocess,
        size=size,
        temperature=temperature,
        name=os.getenv("MODEL_NAME", default_name),
    )
