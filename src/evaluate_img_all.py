#!/usr/bin/env python3

import argparse
import json
import math
import csv
from tempfile import TemporaryDirectory
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image
import torch
from src.my_utils.data_utils import TASKS, images, matching_reference


def read_rgb(path, device):
    with Image.open(path) as image:
        image = np.array(image.convert("RGB"))
    return (
        torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255
    )


def json_number(value):
    return value if math.isfinite(value) else str(value)


def evaluate_fid(metric, records):
    if len(records) < 2:
        raise ValueError("FID requires at least two prediction/GT pairs per task")
    with TemporaryDirectory(prefix="irdiff_fid_") as temporary:
        prediction_dir, reference_dir = (
            Path(temporary) / "prediction",
            Path(temporary) / "reference",
        )
        prediction_dir.mkdir()
        reference_dir.mkdir()
        for index, record in enumerate(records):
            for directory, key in (
                (prediction_dir, "output"),
                (reference_dir, "reference"),
            ):
                source = Path(record[key]).resolve(strict=True)
                (directory / f"{index:08d}{source.suffix}").symlink_to(source)
        return float(
            metric(
                str(prediction_dir),
                str(reference_dir),
                mode="clean",
                num_workers=0,
                batch_size=4,
            )
        )


@torch.inference_mode()
def evaluate_records(records, output, device="cuda", metric_names=("psnr", "ssim")):
    import pyiqa

    if "inception_score" in metric_names:
        raise ValueError("inception_score is not supported by this entry")
    if not records:
        raise ValueError("No prediction/GT pairs found")
    if "fid" in metric_names:
        for task in {record["task"] for record in records}:
            if sum(record["task"] == task for record in records) < 2:
                raise ValueError(
                    f"FID requires at least two prediction/GT pairs for {task}"
                )
    if any(not record.get("reference") for record in records) and "fid" in metric_names:
        raise ValueError("FID requires GT for every selected prediction")
    metrics = {
        name: pyiqa.create_metric(
            name,
            device=device,
            **({"test_y_channel": False} if name in ("psnr", "ssim") else {}),
        )
        for name in metric_names
    }
    per_image_metrics = {
        name: metric for name, metric in metrics.items() if name != "fid"
    }
    requires_reference = any(metric.metric_mode == "FR" for metric in metrics.values())
    results, by_task, task_records = [], {}, {}
    for index, record in enumerate(records, 1):
        if requires_reference and not record.get("reference"):
            raise ValueError(f"GT missing for {record['output']}")
        prediction = read_rgb(record["output"], device)
        reference = (
            read_rgb(record["reference"], device) if requires_reference else None
        )
        if reference is not None and prediction.shape != reference.shape:
            raise ValueError(
                f"Prediction/GT size mismatch for {record['output']}: {prediction.shape} vs {reference.shape}"
            )
        values = {
            name: float(
                (
                    metric(prediction, reference)
                    if metric.metric_mode == "FR"
                    else metric(prediction)
                ).item()
            )
            for name, metric in per_image_metrics.items()
        }
        if any(math.isnan(value) for value in values.values()):
            raise ValueError(f"NaN metric for {record['output']}")
        results.append(
            {
                "output": record["output"],
                "reference": record.get("reference"),
                "task": record["task"],
                **{k: json_number(v) for k, v in values.items()},
            }
        )
        by_task.setdefault(record["task"], []).append(values)
        task_records.setdefault(record["task"], []).append(record)
        print(
            f"[{index}/{len(records)}] {record['task']}: {Path(record['output']).name} "
            + " ".join(f"{k.upper()}={v:.6f}" for k, v in values.items()),
            flush=True,
        )
    tasks = {
        task: {
            "count": len(rows),
            **{k: float(np.mean([r[k] for r in rows])) for k in per_image_metrics},
        }
        for task, rows in by_task.items()
    }
    if "fid" in metrics:
        for task, rows in task_records.items():
            tasks[task]["fid"] = evaluate_fid(metrics["fid"], rows)
            if not math.isfinite(tasks[task]["fid"]):
                raise ValueError(f"Non-finite FID for {task}")
    task_mean = {k: float(np.mean([row[k] for row in tasks.values()])) for k in metrics}
    image_mean = {
        k: float(np.mean([row[k] for rows in by_task.values() for row in rows]))
        for k in per_image_metrics
    }
    report = {
        "protocol": "pyiqa, saved 8-bit RGB PNGs in [0,1]; no external crop or resize; metric-specific preprocessing uses pyiqa defaults",
        "tasks": {
            t: {k: json_number(v) if isinstance(v, float) else v for k, v in r.items()}
            for t, r in tasks.items()
        },
        "task_mean": {k: json_number(v) for k, v in task_mean.items()},
        "image_mean": {k: json_number(v) for k, v in image_mean.items()},
        "images": results,
    }
    if "fid" in metrics:
        report["fid_protocol"] = {
            "mode": "clean",
            "backbone": "inceptionv3",
            "dimensions": 2048,
            "aggregation": "one FID per task over selected prediction/GT sets; task_mean averages task FIDs; image_mean excludes FID",
        }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    with output.with_suffix(".csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["task", "count", *metrics])
        writer.writeheader()
        for task, row in report["tasks"].items():
            writer.writerow({"task": task, **row})
        writer.writerow({"task": "task_mean", **report["task_mean"]})
        writer.writerow({"task": "image_mean", **report["image_mean"]})
    for task, row in tasks.items():
        print(
            f"{task:12s} n={row['count']:5d} "
            + " ".join(f"{k.upper()}={row[k]:.6f}" for k in metrics),
            flush=True,
        )
    print("Task mean: " + " ".join(f"{k.upper()}={task_mean[k]:.6f}" for k in metrics))
    print(f"Metrics saved to {output}")
    return report


def get_images_dict(folder_path):
    root = Path(folder_path)
    return {str(path.relative_to(root)): path for path in images(root)}


def evaluate(
    in_path,
    ref_path=None,
    ntest=None,
    device="cuda",
    metric_names=("psnr", "ssim"),
    output=None,
):
    root = Path(in_path).resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)
    manifest = root / "run_manifest.json"
    if ref_path is None and manifest.is_file():
        records = json.loads(manifest.read_text())["images"]
    else:
        gt_root = Path(ref_path).resolve() if ref_path else None
        records = []
        for relative, path in get_images_dict(root).items():
            relative = Path(relative)
            task = (
                relative.parts[0]
                if relative.parts[0] in TASKS
                else root.name
                if root.name in TASKS
                else "custom"
            )
            if (
                gt_root is not None
                and task in TASKS
                and (gt_root / TASKS[task] / "GT").is_dir()
            ):
                gt_dir = gt_root / TASKS[task] / "GT"
                prediction_dir = root / task if relative.parts[0] == task else root
            else:
                gt_dir, prediction_dir = gt_root, root
            reference = (
                matching_reference(path, prediction_dir, gt_dir) if gt_dir else None
            )
            records.append(
                {
                    "output": str(path),
                    "reference": str(reference) if reference else None,
                    "task": task,
                }
            )
    if ntest is not None:
        counts, selected = {}, []
        for row in records:
            counts[row["task"]] = counts.get(row["task"], 0) + 1
            if counts[row["task"]] <= ntest:
                selected.append(row)
        records = selected
    if not records:
        raise ValueError("No prediction/GT pairs found")
    return evaluate_records(
        records, output or root / "metrics.json", device, metric_names
    )


def main():
    parser = argparse.ArgumentParser(description="Evaluate restored images")
    parser.add_argument(
        "output_dir",
        nargs="?",
        type=Path,
        help="Restored image directory",
    )
    parser.add_argument(
        "-i",
        "--in_path",
        type=Path,
        help="Original interface: restored image directory",
    )
    parser.add_argument(
        "-r", "--ref_path", type=Path, help="GT directory or all_in_one validation root"
    )
    parser.add_argument(
        "--ntest", type=int, default=None, help="Maximum images per task"
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["psnr", "ssim"],
        help="pyiqa metrics, e.g. psnr ssim lpips fid clipiqa niqe musiq arniqa; FID needs >=2 pairs per task",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    path = args.in_path or args.output_dir
    if path is None:
        parser.error("Supply -i / --in_path")
    if args.ntest is not None and args.ntest < 1:
        parser.error("--ntest must be positive")
    evaluate(path, args.ref_path, args.ntest, args.device, args.metrics, args.output)


if __name__ == "__main__":
    main()
