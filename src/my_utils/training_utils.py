"""Paired-training arguments, using the original names and YAML configuration."""
import argparse
import os
from pathlib import Path

import torch
from omegaconf import OmegaConf

from src.my_utils.data_utils import TASKS
from loc_utils.tile_utils import DEFAULT_PROMPT

ROOT = Path(__file__).resolve().parents[2]


def parse_args_paired_training(input_args=None):
    parser = argparse.ArgumentParser(description="Train the restoration-only S3Diff deg/vq model")
    parser.add_argument("--base_config", "--config", dest="base_config", type=Path, default=ROOT / "configs/ir.yaml")
    parser.add_argument("--output_dir", "--output", dest="output_dir", type=Path)
    parser.add_argument("--pretrained_path", "--checkpoint", dest="pretrained_path", type=Path)
    for name in ("sd_path", "daclip_path", "clip_path", "vgg_path", "lpips_path", "train_root", "val_root"):
        parser.add_argument("--" + name, "--" + name.replace("_", "-"), dest=name, type=Path)
    parser.add_argument("--pretrained_path_cls", type=Path)
    parser.add_argument("--pretrained_path_disc", type=Path)
    parser.add_argument("--lora_rank_unet_deg", type=int)
    parser.add_argument("--lora_rank_vae_deg", type=int)
    parser.add_argument("--lora_rank_unet_vq", type=int)
    parser.add_argument("--lora_rank_vae_vq", type=int)
    parser.add_argument("--selector_rank_unet", type=int)
    parser.add_argument("--selector_rank_vae", type=int)
    parser.add_argument("--train_batch_size", "--batch-size", dest="train_batch_size", type=int)
    parser.add_argument("--max_train_steps", "--max-steps", dest="max_train_steps", type=int)
    parser.add_argument("--resolution", "--patch-size", dest="resolution", type=int)
    parser.add_argument("--gradient_accumulation_steps", "--gradient-accumulation", dest="gradient_accumulation_steps", type=int)
    parser.add_argument("--dataloader_num_workers", "--workers", dest="dataloader_num_workers", type=int)
    parser.add_argument("--learning_rate", "--learning-rate", dest="learning_rate", type=float)
    parser.add_argument("--mixed_precision", "--precision", dest="mixed_precision", choices=["no", "fp32", "fp16", "bf16"])
    parser.add_argument("--checkpointing_steps", "--checkpoint-every", dest="checkpointing_steps", type=int)
    parser.add_argument("--eval_freq", "--validate-every", dest="eval_freq", type=int)
    parser.add_argument("--validation_mode", choices=["deg", "deg_vq"], help="Adapter mode used for validation")
    parser.add_argument("--gradient_checkpointing", dest="gradient_checkpointing", action="store_true")
    parser.add_argument("--no-gradient-checkpointing", dest="gradient_checkpointing", action="store_false")
    parser.add_argument("--limit_per_task", "--limit-per-task", dest="limit_per_task", type=int, default=0)
    parser.add_argument("--check_only", "--audit_only", "--audit-only", dest="check_only", action="store_true",
                        help="Verify fresh initialization (or resume weights) and training phases, then exit")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--paired_noise_jpeg", "--paired-noise-jpeg", dest="paired_noise_jpeg", action="store_true")
    parser.add_argument("--pos_prompt", "--prompt", dest="pos_prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--tasks", nargs="+", choices=list(TASKS))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", default="cuda")
    for loss in ("l1", "lpips", "cls", "contra", "gan"):
        parser.add_argument("--lambda_" + loss, "--lambda-" + loss, dest="lambda_" + loss, type=float)
    parser.add_argument("--lr_scheduler", default="cosine", choices=["constant", "constant_with_warmup", "linear", "cosine", "cosine_with_restarts", "polynomial"])
    parser.add_argument("--lr_warmup_steps", type=int, default=-1)
    parser.add_argument("--lr_num_cycles", type=int, default=1)
    parser.add_argument("--lr_power", type=float, default=0.1)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=0.01)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true")
    parser.add_argument("--allow_tf32", action="store_true")
    parser.add_argument("--use_contrastive_loss", action="store_true")
    known, _ = parser.parse_known_args(input_args)
    config = OmegaConf.to_container(OmegaConf.load(known.base_config), resolve=True)
    defaults = dict(config["train_options"])
    defaults.update(config["model_options"])
    defaults.setdefault("validation_mode", "deg_vq" if config["test_options"].get("use_vq", False) else "deg")
    for key in ("output_dir", "pretrained_path", "sd_path", "daclip_path", "clip_path", "vgg_path", "lpips_path"):
        if defaults.get(key) is None:
            continue
        value = Path(defaults[key])
        defaults[key] = str(value if value.is_absolute() else ROOT / value)
    defaults.update(train_root=config["datasets"]["train"]["dataroot"],
                    val_root=config["datasets"]["val"]["dataroot"],
                    tasks=config["datasets"]["train"]["de_type"],
                    resolution=config["datasets"]["train"]["patch_size"])
    parser.set_defaults(**defaults)
    args = parser.parse_args(input_args)
    args.dataset_config = config
    for key in ("output_dir", "pretrained_path", "sd_path", "daclip_path", "clip_path", "vgg_path", "lpips_path", "train_root", "val_root"):
        if getattr(args, key) is not None:
            setattr(args, key, Path(getattr(args, key)).resolve())
    if any((args.pretrained_path, args.pretrained_path_cls, args.pretrained_path_disc)):
        parser.error("Training initializes fresh adapters on SD-Turbo; restoration/prototype/discriminator checkpoints are not accepted. Use --resume for an interrupted training run.")
    args.mixed_precision_requested = args.mixed_precision
    if args.mixed_precision is None:
        args.mixed_precision = os.environ.get("ACCELERATE_MIXED_PRECISION", "no")
    if args.mixed_precision == "fp32":
        args.mixed_precision = "no"
    if args.mixed_precision not in ("no", "fp16", "bf16"):
        parser.error("Invalid ACCELERATE_MIXED_PRECISION; use no, fp16 or bf16")
    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        parser.error("Use a CUDA-enabled PyTorch environment and --device cuda[:N]")
    if min(args.max_train_steps, args.train_batch_size, args.gradient_accumulation_steps) < 1:
        parser.error("steps, batch size and gradient accumulation must be positive")
    if min(args.dataloader_num_workers, args.limit_per_task, args.checkpointing_steps, args.eval_freq) < 0:
        parser.error("workers, data limits and checkpoint/validation intervals must be non-negative")
    if args.resolution < 32 or args.resolution % 32:
        parser.error("--resolution must be a multiple of 32, at least 32")
    if len(args.tasks) != len(set(args.tasks)):
        parser.error("Duplicate tasks")
    if args.lambda_l1 <= 0 and args.lambda_gan <= 0:
        parser.error("The vq phase needs lambda_l1 or lambda_gan > 0")
    if any(getattr(args, "lambda_" + loss) < 0 for loss in ("l1", "lpips", "cls", "contra", "gan")):
        parser.error("Loss weights must be non-negative")
    return args
