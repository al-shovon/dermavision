"""Skin-lesion classifier wrapper.

The web app only ever calls .predict(). To replace the pretrained model with your own
later, change the environment variables read in model_loader.py (see .env.example).
No other code needs to change.
"""
import timm
import torch
from PIL import Image
from torchvision import transforms as T

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(mode: str = "squash", size: int = 224) -> T.Compose:
    """'squash' resizes the whole image to size x size.
    'crop' resizes to size / 0.875 and then centre-crops, like timm's default."""
    bicubic = T.InterpolationMode.BICUBIC
    if mode == "squash":
        ops = [T.Resize((size, size), interpolation=bicubic)]
    elif mode == "crop":
        ops = [T.Resize(int(size / 0.875), interpolation=bicubic), T.CenterCrop(size)]
    else:
        raise ValueError(f"Unknown PREPROCESS mode: {mode!r} (use 'squash' or 'crop')")
    return T.Compose(ops + [T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])


def build_model(arch: str, num_classes: int):
    """'efficientnet_b0' builds the timm model; 'tv_efficientnet_b0' builds the torchvision one."""
    if arch.startswith("tv_"):
        import torchvision

        return getattr(torchvision.models, arch[3:])(weights=None, num_classes=num_classes)
    return timm.create_model(arch, pretrained=False, num_classes=num_classes)


def find_tensors(obj):
    """Find the flat {layer name: tensor} dict inside a checkpoint, however it is wrapped."""
    if isinstance(obj, dict):
        if obj and all(isinstance(v, torch.Tensor) for v in obj.values()):
            return obj
        for v in obj.values():
            found = find_tensors(v)
            if found:
                return found
    return None


def strip_prefixes(state: dict) -> dict:
    """Drop wrapper prefixes such as DataParallel's 'module.' or torch.compile's '_orig_mod.'."""
    for prefix in ("module.", "_orig_mod.", "model.", "backbone."):
        if state and all(k.startswith(prefix) for k in state):
            state = {k[len(prefix):]: v for k, v in state.items()}
    return state


def read_state_dict(path: str, model) -> dict:
    # weights_only=True refuses to run arbitrary code stored in a checkpoint.
    state = strip_prefixes(find_tensors(torch.load(path, map_location="cpu", weights_only=True)) or {})
    if "classifier.1.weight" in state and "classifier.weight" in model.state_dict():
        state = {k.replace("classifier.1.", "classifier."): v for k, v in state.items()}
    return state


class SkinClassifier:
    def __init__(self, weights_path, labels, arch="efficientnet_b0", preprocess="squash",
                 size=224, temperature=1.0, name="model"):
        self.name = name
        self.labels = list(labels)
        self.temperature = float(temperature)
        self.transform = build_transform(preprocess, size)
        self.model = build_model(arch, len(self.labels))
        self.model.load_state_dict(read_state_dict(weights_path, self.model))
        self.model.eval()

    @torch.inference_mode()
    def predict(self, img: Image.Image) -> dict:
        """Return {class_code: probability} for one PIL image."""
        x = self.transform(img.convert("RGB")).unsqueeze(0)
        probs = torch.softmax(self.model(x) / self.temperature, dim=1)[0].tolist()
        return dict(zip(self.labels, probs))
