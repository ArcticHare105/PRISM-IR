"""Load LPIPS and its VGG trunk explicitly from local files, without downloads."""
import torch


def load_lpips(vgg_path, calibration_path, device):
    import lpips
    model = lpips.LPIPS(net="vgg", pnet_rand=True, model_path=str(calibration_path))
    vgg = torch.load(vgg_path, map_location="cpu")
    trunk = {name: vgg["features." + name.split(".", 1)[1]] for name in model.net.state_dict()}
    model.net.load_state_dict(trunk, strict=True)
    return model.to(device).eval().requires_grad_(False)


import json
from pathlib import Path
from src.open_clip.model import CLIP
from src.open_clip.daclip_model import DaCLIP


def load_restoration_state(module, state, required):
    current = module.state_dict()
    unexpected = set(state) - set(current)
    missing = {key for key in current if required(key)} - set(state)
    if unexpected or missing:
        raise ValueError(f"Invalid restoration checkpoint: unexpected={sorted(unexpected)}, missing={sorted(missing)}")
    current.update(state)
    module.load_state_dict(current, strict=True)


def load_daclip(checkpoint, device="cuda"):
    config = json.loads((Path(__file__).resolve().parents[1] / "open_clip/model_configs/daclip_ViT-B-32.json").read_text())
    config.pop("custom_text", None)
    model = DaCLIP(CLIP(**config))
    state = torch.load(checkpoint, map_location="cpu")
    state = state.get("state_dict", state)
    state = {k.removeprefix("module."): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    return model.to(device).eval().requires_grad_(False)
