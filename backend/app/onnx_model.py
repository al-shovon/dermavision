"""ONNX Runtime version of the classifier.

It needs no PyTorch, so the production image stays small and fits free hosting
(for example a 512 MB instance). Training and Grad-CAM still use PyTorch.
"""
import os

import numpy as np
from PIL import Image

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess(img: Image.Image, size: int = 224) -> np.ndarray:
    """Same steps as the PyTorch 'squash' pipeline: resize, scale to 0-1, normalise, NCHW."""
    img = img.convert("RGB").resize((size, size), Image.Resampling.BICUBIC)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    return arr.transpose(2, 0, 1)[None, ...]


class OnnxClassifier:
    def __init__(self, model_path, labels, preprocess_mode="squash", size=224, temperature=1.0, name="onnx-model"):
        if preprocess_mode != "squash":
            raise ValueError("The ONNX classifier supports PREPROCESS=squash only.")
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = int(os.getenv("ORT_THREADS", "1"))
        self.session = ort.InferenceSession(
            str(model_path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self.labels = list(labels)
        self.size = size
        self.temperature = float(temperature)
        self.name = name

    def predict(self, img: Image.Image) -> dict:
        """Return {class_code: probability} for one PIL image."""
        logits = self.session.run(None, {self.input_name: preprocess(img, self.size)})[0][0]
        if len(logits) != len(self.labels):
            raise RuntimeError(f"The model returns {len(logits)} scores but {len(self.labels)} labels are set.")
        z = logits.astype(np.float64) / self.temperature
        z -= z.max()
        probs = np.exp(z)
        probs /= probs.sum()
        return dict(zip(self.labels, probs.tolist()))
