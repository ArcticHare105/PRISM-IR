import json
from pathlib import Path
import random

import numpy as np
import torch

from src.open_clip.model import CLIP
from src.open_clip.daclip_model import DaCLIP


def set_seed(seed):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def resolve_sd_path(path=None):
    if path is None:
        from huggingface_hub import snapshot_download

        path = snapshot_download(
            repo_id="stabilityai/sd-turbo",
            revision="b261bac6fd2cf515557d5d0707481eafa0485ec2",
            allow_patterns=[
                "model_index.json",
                "scheduler/*",
                "tokenizer/*",
                "text_encoder/config.json",
                "text_encoder/model.safetensors",
                "unet/config.json",
                "unet/diffusion_pytorch_model.safetensors",
                "vae/config.json",
                "vae/diffusion_pytorch_model.safetensors",
            ],
        )
    return Path(path).expanduser().resolve()


def load_lpips(vgg_path=None, calibration_path=None, device="cuda"):
    import lpips
    from torchvision.models import VGG16_Weights

    model = lpips.LPIPS(
        net="vgg",
        pnet_rand=True,
        model_path=str(calibration_path) if calibration_path is not None else None,
    )
    vgg = (
        torch.load(vgg_path, map_location="cpu")
        if vgg_path is not None
        else VGG16_Weights.IMAGENET1K_V1.get_state_dict(progress=True, check_hash=True)
    )
    trunk = {
        name: vgg["features." + name.split(".", 1)[1]]
        for name in model.net.state_dict()
    }
    model.net.load_state_dict(trunk, strict=True)
    return model.to(device).eval().requires_grad_(False)


def load_restoration_state(module, state, required):
    current = module.state_dict()
    unexpected = set(state) - set(current)
    missing = {key for key in current if required(key)} - set(state)
    if unexpected or missing:
        raise ValueError(
            f"Invalid restoration checkpoint: unexpected={sorted(unexpected)}, missing={sorted(missing)}"
        )
    current.update(state)
    module.load_state_dict(current, strict=True)


def load_daclip(checkpoint, device="cuda"):
    config = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "open_clip/model_configs/daclip_ViT-B-32.json"
        ).read_text()
    )
    config.pop("custom_text", None)
    model = DaCLIP(CLIP(**config))
    state = torch.load(checkpoint, map_location="cpu")
    state = state.get("state_dict", state)
    state = {k.removeprefix("module."): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    return model.to(device).eval().requires_grad_(False)
