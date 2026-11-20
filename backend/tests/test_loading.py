"""The loader must cope with the different ways a checkpoint can be saved."""
import pytest
import timm
import torch
from PIL import Image

from app.skin_model import SkinClassifier

LABELS = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]


@pytest.mark.parametrize("layout", ["plain", "module", "compiled", "nested"])
def test_checkpoint_layouts(tmp_path, layout):
    state = timm.create_model("efficientnet_b0", pretrained=False, num_classes=7).state_dict()
    if layout == "module":
        state = {"module." + k: v for k, v in state.items()}
    elif layout == "compiled":
        state = {"_orig_mod." + k: v for k, v in state.items()}
    elif layout == "nested":
        state = {"epoch": 3, "model_state_dict": state}
    path = tmp_path / "weights.pth"
    torch.save(state, path)

    clf = SkinClassifier(path, LABELS)
    probs = clf.predict(Image.new("RGB", (80, 60), (180, 120, 100)))
    assert list(probs) == LABELS
    assert abs(sum(probs.values()) - 1) < 1e-4
