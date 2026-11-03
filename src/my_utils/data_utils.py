"""File discovery and preprocessing for restoration tests."""
from dataclasses import dataclass
from pathlib import Path
import zlib

import numpy as np
from PIL import Image
import torch


TASKS = {
    "denoise": "noisy", "derain": "rainy", "dehaze": "hazy",
    "delowlight": "low-light", "deblur": "motion-blurry", "desnow": "snowy",
    "deraindrop": "raindrop", "deshadow": "shadowed", "dejpeg": "jpeg-compressed",
    "inpaint": "uncompleted",
}
EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass
class Sample:
    source: Path
    output: Path
    task: str = "custom"
    reference: Path = None


def images(root):
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in EXTENSIONS)


def matching_reference(source, lq_root, gt_root):
    relative = source.relative_to(lq_root)
    exact = gt_root / relative
    if exact.is_file():
        return exact
    matches = [p for p in exact.parent.glob(relative.stem + ".*") if p.suffix.lower() in EXTENSIONS]
    # ISTD/deshadow GTs in all_in_one use *_free.png, unlike their LQ names.
    if not matches and gt_root.parent.name == "shadowed":
        matches = [p for p in exact.parent.glob(relative.stem + "_free.*") if p.suffix.lower() in EXTENSIONS]
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one GT image for {source}, found {matches}")
    return matches[0]


def discover(input_path, dataset_root, output, tasks, limit, gt_dir=None):
    records = []
    if input_path is not None:
        input_path = Path(input_path).resolve()
        if not input_path.exists():
            raise FileNotFoundError(input_path)
        lq_root = input_path.parent if input_path.is_file() else input_path
        paths = [input_path] if input_path.is_file() else images(input_path)
        if limit:
            paths = paths[:limit]
        for p in paths:
            reference = matching_reference(p, lq_root, Path(gt_dir)) if gt_dir else None
            records.append(Sample(p, output / p.relative_to(lq_root).with_suffix(".png"), reference=reference))
    else:
        dataset_root = Path(dataset_root).resolve()
        for task in tasks:
            root = dataset_root / TASKS[task]
            lq_root, gt_root = root / "LQ", root / "GT"
            if not lq_root.is_dir() or not gt_root.is_dir():
                raise FileNotFoundError(f"Missing LQ/GT directories under {root}")
            paths = images(lq_root)
            if not paths:
                raise ValueError(f"No images in {lq_root}")
            if limit:
                paths = paths[:limit]
            for p in paths:
                records.append(Sample(p, output / task / p.relative_to(lq_root).with_suffix(".png"), task,
                                      matching_reference(p, lq_root, gt_root)))
    if not records:
        raise ValueError("No input images found")
    outputs = [str(s.output) for s in records]
    if len(outputs) != len(set(outputs)):
        raise ValueError("Input images map to duplicate PNG output paths")
    for sample in records:
        if sample.output.resolve() == sample.source.resolve() or (
                sample.reference and sample.output.resolve() == sample.reference.resolve()):
            raise ValueError("Output path must differ from input and GT")
    return records


def load_input(path):
    array = np.asarray(Image.open(path).convert("RGB"))
    height, width = array.shape[:2]
    # OpenCV BORDER_REFLECT in the original loader includes the border pixel.
    array = np.pad(array, ((0, (-height) % 32), (0, (-width) % 32), (0, 0)), mode="symmetric")
    tensor = torch.from_numpy(array.copy()).permute(2, 0, 1).float().div_(255).mul_(2).sub_(1).unsqueeze(0)
    return tensor, (height, width)


def image_seed(base, relative_path):
    return (base + zlib.crc32(str(relative_path).encode())) % (2**32)
