#!/usr/bin/env python3
"""Run the independent image restoration test pipeline."""
import gc
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import random

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
from PIL import Image
import torch

from src.my_utils.data_utils import image_seed
from src.my_utils.utils import load_daclip
from src.s3diff import S3Diff
from src.my_utils.testing_utils import parse_args_paired_testing
from src.my_utils.test_dataset_utils import IRTestDataset
from loc_utils.tile_utils import RestorationPipeline
from src.my_utils.model_checks import model_structure


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main(args):
    output = args.output_dir.resolve()
    dataset_val = IRTestDataset(args.dataset_config["datasets"]["val"], args=args)
    samples = dataset_val.samples
    # Reject a recursive input/output layout which could ingest previous outputs.
    if args.input and args.input.is_dir() and output.is_relative_to(args.input.resolve()):
        raise ValueError("Output directory must be outside the input directory")
    if args.evaluate:
        import pyiqa  # Fail before a long inference run if evaluation dependencies are missing.
    for path in (args.sd_path, args.daclip_path, args.pretrained_path):
        if not path.exists():
            raise FileNotFoundError(path)
    seed_everything(args.seed)
    torch.backends.cudnn.benchmark = False
    net_sr = S3Diff(sd_path=str(args.sd_path), pretrained_path=args.pretrained_path, device=args.device,
                    lora_rank_unet_deg=args.lora_rank_unet_deg, lora_rank_vae_deg=args.lora_rank_vae_deg,
                    lora_rank_unet_vq=args.lora_rank_unet_vq, lora_rank_vae_vq=args.lora_rank_vae_vq,
                    load_vq=args.use_vq)
    net_sr.set_lora(classify=not args.use_vq)
    net_sr.set_eval()
    net_sr.requires_grad_(False)
    for network in ("unet", "vae"):
        for adapter in ("deg", "vq"):
            setattr(args, f"lora_rank_{network}_{adapter}", getattr(net_sr, f"lora_rank_{network}_{adapter}"))
    adapter_state = {
        name: {"loaded_adapters": sorted(network.peft_config),
               "active_adapters": sorted({adapter for module in network.modules() if hasattr(module, "lora_A")
                                          for adapter in module.active_adapters})}
        for name, network in (("unet", net_sr.unet), ("vae", net_sr.vae))}
    print(f"Inference adapters: {json.dumps(adapter_state)}", flush=True)
    if args.enable_xformers_memory_efficient_attention:
        net_sr.unet.enable_xformers_memory_efficient_attention()
    daclip = load_daclip(args.daclip_path, args.device)
    pipeline = RestorationPipeline(net_sr, daclip, args.pos_prompt, args.tile_size, args.tile_overlap)
    dtype = {"no": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[args.mixed_precision]
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for index, sample in enumerate(samples, 1):
        seed = image_seed(args.seed, sample.output.relative_to(output))
        skipped = sample.output.exists() and not args.overwrite
        if not skipped:
            seed_everything(seed)
            batch_val = dataset_val[index - 1]
            tensor = batch_val["lq"].unsqueeze(0)
            height, width = batch_val["ori_h"], batch_val["ori_w"]
            with torch.autocast(device_type=torch.device(args.device).type, dtype=dtype, enabled=args.mixed_precision != "no"):
                prediction = pipeline(tensor)[0, :, :height, :width]
            if not torch.isfinite(prediction).all():
                raise ValueError(f"Non-finite output for {sample.source}; use --precision fp32")
            array = prediction.add(1).mul(0.5).clamp(0, 1).mul(255).byte().permute(1, 2, 0).numpy()
            sample.output.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(array).save(sample.output)
        record = {"input": str(sample.source), "output": str(sample.output), "task": sample.task,
                  "reference": str(sample.reference) if sample.reference else None,
                  "seed": seed, "skipped": skipped}
        records.append(record)
        print(f"[{index}/{len(samples)}] {sample.task}: {sample.source.name} -> {sample.output} "
              + ("(exists, skipped)" if skipped else ""), flush=True)
    report = {"settings": vars(args), "model": model_structure(net_sr), "adapters": adapter_state, "images": records}
    manifest_path = args.manifest_path or output / "run_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=os.fspath) + "\n")
    if args.evaluate:
        del pipeline, net_sr, daclip
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        from src.evaluate_img_all import evaluate_records
        evaluate_records(records, output / "metrics.json", args.device)


if __name__ == "__main__":
    args = parse_args_paired_testing()
    main(args)
