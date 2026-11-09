import numpy as np
import torch
from torch.nn import functional as F


DEFAULT_PROMPT = "A high-resolution, 8K, ultra-realistic image with sharp focus, vibrant colors, and natural lighting."


def tile_starts(length, size, overlap):
    starts = list(range(0, max(length - size, 0) + 1, size - overlap))
    end = max(length - size, 0)
    if starts[-1] != end:
        starts.append(end)
    return starts


def gaussian_weights(height, width):
    x = np.arange(width, dtype=np.float64) - (width - 1) / 2
    y = np.arange(height, dtype=np.float64) - height / 2
    x = np.exp(-x * x / (width * width) / 0.02) / np.sqrt(2 * np.pi * 0.01)
    y = np.exp(-y * y / (height * height) / 0.02) / np.sqrt(2 * np.pi * 0.01)
    return torch.tensor(np.outer(y, x), dtype=torch.float32)[None, None]


class RestorationPipeline:
    def __init__(
        self, model, daclip, prompt=DEFAULT_PROMPT, tile_size=512, tile_overlap=128
    ):
        if tile_size and (tile_size < 32 or tile_size % 32):
            raise ValueError("tile_size must be 0 or a positive multiple of 32")
        if (
            tile_overlap < 0
            or tile_overlap % 32
            or (tile_size and tile_overlap >= tile_size)
        ):
            raise ValueError(
                "tile_overlap must be a multiple of 32, smaller than tile_size"
            )
        self.model, self.daclip = model, daclip
        self.prompt, self.tile_size, self.tile_overlap = prompt, tile_size, tile_overlap
        self.device = next(model.parameters()).device
        self.mean = torch.tensor(
            [0.48145466, 0.4578275, 0.40821073], device=self.device
        )[None, :, None, None]
        self.std = torch.tensor(
            [0.26862954, 0.26130258, 0.27577711], device=self.device
        )[None, :, None, None]

    def restore_patch(self, image):
        features = F.interpolate(
            image * 0.5 + 0.5, size=(224, 224), mode="bilinear", align_corners=True
        )
        features = (features - self.mean) / self.std
        image_context, degradation_context, _ = self.daclip.encode_image(
            features, control=True
        )
        return self.model(
            image,
            image_context.float(),
            degradation_context.float(),
            [self.prompt] * len(image),
        )

    @torch.inference_mode()
    def __call__(self, image):
        _, _, height, width = image.shape
        if not self.tile_size or max(height, width) <= self.tile_size:
            return self.restore_patch(image.to(self.device)).float().cpu()

        image = image.cpu()
        th, tw = min(height, self.tile_size), min(width, self.tile_size)
        oy = min(self.tile_overlap, th - 32)
        ox = min(self.tile_overlap, tw - 32)
        weights = gaussian_weights(th, tw)
        result = torch.zeros_like(image, dtype=torch.float32)
        contributors = torch.zeros((1, 1, height, width), dtype=torch.float32)
        for y in tile_starts(height, th, oy):
            for x in tile_starts(width, tw, ox):
                patch = image[:, :, y : y + th, x : x + tw].to(self.device)
                prediction = self.restore_patch(patch).float().cpu()
                result[:, :, y : y + th, x : x + tw] += prediction * weights
                contributors[:, :, y : y + th, x : x + tw] += weights
        return result / contributors
