from io import BytesIO
from pathlib import Path
import random

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from src.my_utils.data_utils import TASKS, images

ROOT = Path(__file__).resolve().parents[2]


TASK_WEIGHTS = {
    "denoise": 1,
    "derain": 7,
    "deraindrop": 5,
    "dehaze": 5,
    "deblur": 8,
    "delowlight": 3,
    "desnow": 3,
    "deshadow": 3,
    "dejpeg": 2,
    "inpaint": 3,
}


def find_gt(path, lq_root, gt_root, task):
    relative = path.relative_to(lq_root)
    candidate = gt_root / relative
    if candidate.is_file():
        return candidate
    stem = relative.stem
    if task == "derain" and stem.startswith("rain-"):
        stem = "norain-" + stem[5:]
    elif task == "deraindrop" and stem.endswith("_rain"):
        stem = stem[:-5] + "_clean"
    elif task == "deshadow":
        stem += "_no_shadow"
    matches = [
        p
        for p in (gt_root / relative.parent).glob(stem + ".*")
        if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    ]
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected exactly one GT for {path}, got {matches}")
    return matches[0]


class IRTrainDataset(Dataset):
    def __init__(self, dataset_opt):
        root = dataset_opt.get("dataroot", ROOT)
        tasks = dataset_opt["de_type"]
        patch_size = dataset_opt.get("patch_size", 256)
        samples = dataset_opt.get("samples", 10000)
        limit = dataset_opt.get("limit_per_task", 0)
        synthetic_noise_jpeg = dataset_opt.get("synthetic_noise_jpeg", True)
        label_smoothing = dataset_opt.get("label_smooth", 0.1)
        self.dataset_opt = dataset_opt
        self.root, self.tasks = Path(root), tasks
        self.patch_size, self.samples = patch_size, samples
        self.synthetic = synthetic_noise_jpeg
        self.smoothing = label_smoothing
        self.pairs = {}
        for task in tasks:
            group_files = dataset_opt.get(task + "_file")
            if group_files:
                pairs = []
                for group_file in group_files:
                    group_file = Path(group_file)
                    if not group_file.is_absolute():
                        group_file = ROOT / group_file
                    for line in group_file.read_text().splitlines():
                        if not line.strip():
                            continue
                        lq, gt = line.split("|")
                        lq, gt = (
                            Path(lq.strip()) if lq.strip() else Path(gt.strip()),
                            Path(gt.strip()),
                        )
                        if not gt.is_file() or (
                            not lq.is_file() and task not in ("denoise", "dejpeg")
                        ):
                            raise FileNotFoundError(
                                f"Missing training pair: {lq} | {gt}"
                            )
                        pairs.append((lq, gt))
                self.pairs[task] = pairs[:limit] if limit else pairs
                if not self.pairs[task]:
                    raise ValueError(f"No training pairs for {task}")
                continue
            folder = self.root / TASKS[task]
            lq, gt = folder / "LQ", folder / "GT"
            if not lq.is_dir() or not gt.is_dir():
                raise FileNotFoundError(f"Training LQ/GT missing: {folder}")
            paths = images(lq)
            if limit:
                paths = paths[:limit]
            if not paths:
                raise ValueError(f"No training images in {lq}")
            self.pairs[task] = [(p, find_gt(p, lq, gt, task)) for p in paths]

    def __len__(self):
        return self.samples

    def __getitem__(self, index):
        task = random.choices(
            self.tasks, weights=[TASK_WEIGHTS[t] for t in self.tasks]
        )[0]
        lq_path, gt_path = random.choice(self.pairs[task])
        with Image.open(gt_path) as image:
            gt = np.array(image.convert("RGB"))
        if self.synthetic and task in ("denoise", "dejpeg"):
            lq = gt.copy()
        else:
            with Image.open(lq_path) as image:
                lq = np.array(image.convert("RGB"))
        if lq.shape != gt.shape:
            raise ValueError(
                f"Training pair size mismatch: {lq_path} {lq.shape}, {gt_path} {gt.shape}"
            )
        height, width = gt.shape[:2]
        size = self.patch_size
        pad = ((0, max(0, size - height)), (0, max(0, size - width)), (0, 0))
        gt, lq = np.pad(gt, pad, mode="symmetric"), np.pad(lq, pad, mode="symmetric")

        crop = min(600, gt.shape[0], gt.shape[1]) if task == "deshadow" else size
        y, x = (
            random.randint(0, gt.shape[0] - crop),
            random.randint(0, gt.shape[1] - crop),
        )
        gt, lq = gt[y : y + crop, x : x + crop], lq[y : y + crop, x : x + crop]
        if crop != size:
            gt = np.array(
                Image.fromarray(gt).resize((size, size), Image.Resampling.BILINEAR)
            )
            lq = np.array(
                Image.fromarray(lq).resize((size, size), Image.Resampling.BILINEAR)
            )
        rotation, flip = random.randrange(4), random.randrange(2)
        gt, lq = np.rot90(gt, rotation), np.rot90(lq, rotation)
        if flip:
            gt, lq = gt[:, ::-1], lq[:, ::-1]
        if self.synthetic and task == "denoise":
            sigma = random.choice((15, 25, 50))
            noise = np.random.normal(0, sigma, gt.shape)
            lq = np.clip(gt.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        elif self.synthetic and task == "dejpeg":
            with BytesIO() as buffer:
                Image.fromarray(gt).save(buffer, format="JPEG", quality=10)
                buffer.seek(0)
                with Image.open(buffer) as image:
                    lq = np.array(image.convert("RGB"))

        def tensor(array):
            return (
                torch.from_numpy(array.copy())
                .permute(2, 0, 1)
                .float()
                .div(127.5)
                .sub(1)
            )

        label = torch.full((len(self.tasks),), self.smoothing / len(self.tasks))
        label[self.tasks.index(task)] += 1 - self.smoothing
        return {
            "degrad_patch": tensor(lq),
            "clean_patch": tensor(gt),
            "de_label": label,
            "degra": task,
            "input": str(lq_path),
            "reference": str(gt_path),
        }
