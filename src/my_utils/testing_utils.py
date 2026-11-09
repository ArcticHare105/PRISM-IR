import argparse
from pathlib import Path
import torch
from omegaconf import OmegaConf

from src.my_utils.data_utils import TASKS
from loc_utils.tile_utils import DEFAULT_PROMPT

ROOT = Path(__file__).resolve().parents[2]


def parse_args_paired_testing(input_args=None):
    parser = argparse.ArgumentParser(
        description="Test the S3Diff image-restoration branch"
    )
    parser.add_argument(
        "--base_config", type=Path, default=ROOT / "configs/ir_10task.yaml"
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--input", type=Path)
    group.add_argument("--all-in-one", action="store_true")
    group.add_argument(
        "--group_file", nargs="+", type=Path, help="Original LQ|GT test TXT lists"
    )
    parser.add_argument("--output_dir", "--output", dest="output_dir", type=Path)
    parser.add_argument(
        "--manifest_path",
        type=Path,
        help="Separate run manifest for parallel task groups",
    )
    parser.add_argument(
        "--pretrained_path", "--checkpoint", dest="pretrained_path", type=Path
    )
    parser.add_argument("--sd_path", "--sd-path", dest="sd_path", type=Path)
    parser.add_argument("--daclip_path", "--daclip-path", dest="daclip_path", type=Path)
    parser.add_argument(
        "--dataset_root", "--dataset-root", dest="dataset_root", type=Path
    )
    parser.add_argument("--ref_path", "--gt-dir", dest="gt_dir", type=Path)
    parser.add_argument("--tasks", nargs="+", choices=list(TASKS))
    parser.add_argument(
        "--lora_rank_unet_deg",
        "--lora_rank_unet_ir",
        dest="lora_rank_unet_deg",
        type=int,
        help="Default: read from checkpoint",
    )
    parser.add_argument(
        "--lora_rank_vae_deg",
        "--lora_rank_vae_ir",
        dest="lora_rank_vae_deg",
        type=int,
        help="Default: read from checkpoint",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--use_vq",
        "--use-vq",
        action="store_true",
        help="Use deg+vq; requires a full checkpoint (current config default)",
    )
    mode.add_argument(
        "--deg_only",
        "--deg-only",
        dest="use_vq",
        action="store_false",
        help="Use only the degradation adapter",
    )
    parser.add_argument(
        "--lora_rank_unet_vq", type=int, help="Default: read from checkpoint"
    )
    parser.add_argument(
        "--lora_rank_vae_vq", type=int, help="Default: read from checkpoint"
    )
    parser.add_argument(
        "--mixed_precision",
        "--precision",
        dest="mixed_precision",
        choices=["no", "fp32", "fp16", "bf16"],
    )
    parser.add_argument(
        "--tile_size",
        "--tile-size",
        dest="tile_size",
        type=int,
        help="0 for whole-image inference; a positive multiple of 32 enables tiles",
    )
    parser.add_argument(
        "--tile_overlap",
        "--tile-overlap",
        dest="tile_overlap",
        type=int,
        help="Overlap between tiles, in pixels (default: 128)",
    )
    parser.add_argument(
        "--pos_prompt", "--prompt", dest="pos_prompt", default=DEFAULT_PROMPT
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--inference_timestep", type=int, default=300)
    parser.add_argument(
        "--denoising_mode",
        choices=["scheduler", "prediction_minus_latent"],
        default="scheduler",
    )
    parser.add_argument("--limit", "--ntest", dest="limit", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true"
    )
    known, _ = parser.parse_known_args(input_args)
    config = OmegaConf.to_container(OmegaConf.load(known.base_config), resolve=True)
    defaults = dict(config.get("model_options", {}))
    defaults.update(config["test_options"])
    defaults["seed"] = config.get("seed", defaults.get("seed"))
    for key in ("output_dir", "pretrained_path", "sd_path", "daclip_path"):
        if key == "sd_path" and defaults.get(key) is None:
            continue
        value = Path(defaults[key])
        defaults[key] = str(value if value.is_absolute() else ROOT / value)
    defaults.update(
        dataset_root=config["datasets"]["val"]["dataroot"],
        tasks=config["datasets"]["train"]["de_type"],
    )
    parser.set_defaults(**defaults)
    args = parser.parse_args(input_args)
    args.dataset_config = config
    for name in (
        "output_dir",
        "pretrained_path",
        "sd_path",
        "daclip_path",
        "dataset_root",
    ):
        if getattr(args, name) is not None:
            setattr(args, name, Path(getattr(args, name)).resolve())
    if args.mixed_precision == "fp32":
        args.mixed_precision = "no"
    if args.limit < 0:
        parser.error("--limit must be non-negative")
    if args.input and args.evaluate and not args.gt_dir:
        parser.error("--evaluate with --input requires --ref_path / --gt-dir")
    if len(args.tasks) != len(set(args.tasks)):
        parser.error("Duplicate tasks")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("Use a CUDA-enabled PyTorch environment")
    if args.device == "cpu" and args.mixed_precision != "no":
        parser.error("CPU inference requires --mixed_precision no")
    return args
