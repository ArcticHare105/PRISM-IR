#!/usr/bin/env python3
"""Two-phase restoration training, with the original S3Diff file and argument layout."""
from contextlib import nullcontext
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
from torch.nn import functional as F
from torch.utils.data import DataLoader
from diffusers.optimization import get_scheduler

from src.my_utils.data_utils import TASKS, discover, load_input
from src.my_utils.utils import load_daclip
from src.my_utils.model_checks import check_checkpoint, check_fresh_initialization, check_gradients
from src.my_utils.dataset_utils import IRTrainDataset
from src.model import contrastive_select_loss, soft_target_ce_loss, ssim_loss
from src.s3diff import S3Diff
from src.classify_model import proto_DegraCls

from src.my_utils.training_utils import parse_args_paired_training


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main(args):
    torch.set_num_threads(4)
    device = torch.device(args.device)
    if args.seed is not None:
        seed_all(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints = args.output_dir / "checkpoints"
    checkpoints.mkdir(exist_ok=True)
    settings = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    resume = torch.load(args.resume, map_location="cpu") if args.resume else None
    data_seed = resume["settings"]["data_seed"] if resume and "data_seed" in resume["settings"] else torch.initial_seed() % (2**32)
    settings["data_seed"] = data_seed
    initial_checkpoint = None
    if resume:
        # The resume file stores all net_sr tensors. Materialize locally so the
        # same strict net_sr loader is used for initialization and continuation.
        initial_checkpoint = args.output_dir / "resume_initial_model.pkl"
        torch.save(resume["model"], initial_checkpoint)
        aliases = {"train_batch_size": "batch_size", "gradient_accumulation_steps": "gradient_accumulation",
                   "resolution": "patch_size", "mixed_precision": "precision"}
        for key in ("tasks", "seed", "train_batch_size", "gradient_accumulation_steps", "resolution", "paired_noise_jpeg",
                    "mixed_precision", "learning_rate", "lr_scheduler", "lr_warmup_steps", "lr_num_cycles", "lr_power",
                    "lambda_l1", "lambda_lpips", "lambda_cls", "lambda_contra", "use_contrastive_loss", "lambda_gan", "train_root"):
            saved = resume["settings"].get(key, resume["settings"].get(aliases.get(key)))
            if key == "mixed_precision" and saved == "fp32":
                saved = "no"
            if saved != settings[key]:
                raise ValueError(f"Resume setting differs: {key}")
    net_sr = S3Diff(sd_path=str(args.sd_path), pretrained_path=initial_checkpoint,
                    lora_rank_unet_deg=args.lora_rank_unet_deg, lora_rank_vae_deg=args.lora_rank_vae_deg,
                    lora_rank_unet_vq=args.lora_rank_unet_vq, lora_rank_vae_vq=args.lora_rank_vae_vq,
                    selector_rank_unet=args.selector_rank_unet, selector_rank_vae=args.selector_rank_vae,
                    class_num=len(TASKS), device=args.device)
    net_sr.set_train()
    structure = check_checkpoint(net_sr, initial_checkpoint) if resume else check_fresh_initialization(net_sr)
    for network in ("unet", "vae"):
        for adapter in ("deg", "vq"):
            name = f"lora_rank_{network}_{adapter}"
            setattr(args, name, getattr(net_sr, name))
            settings[name] = getattr(net_sr, name)
    settings["model"] = structure
    settings["initial_checkpoint"] = str(args.resume) if args.resume else None
    settings["base_model"] = str(args.sd_path)
    settings["training_mode"] = "resume" if args.resume else "fresh_lora_on_sd_turbo"
    (args.output_dir / "settings.json").write_text(json.dumps(settings, indent=2) + "\n")
    (args.output_dir / "model_structure.json").write_text(json.dumps(structure, indent=2) + "\n")
    if resume:
        initial_checkpoint.unlink()
    print("Resume weights and phases verified:" if resume else "Fresh initialization and phases verified:",
          structure["ranks"], flush=True)
    if args.check_only:
        return
    if args.gradient_checkpointing:
        net_sr.unet.enable_gradient_checkpointing()
    if args.enable_xformers_memory_efficient_attention:
        net_sr.unet.enable_xformers_memory_efficient_attention()
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
    net_proto_cls = proto_DegraCls(net_sr.rank_vae, net_sr.rank_unet, class_num=len(TASKS)).to(device)
    # Use the same parameter ordering for fresh initialization and resume.
    trainable = [p for module in (net_sr.vae, net_sr.unet, net_sr.vae_selector, net_sr.unet_selector)
                 for p in module.parameters() if p.requires_grad] + list(net_proto_cls.parameters())
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, betas=(args.adam_beta1, args.adam_beta2),
                                 weight_decay=args.adam_weight_decay, eps=args.adam_epsilon)
    net_daclip = load_daclip(args.daclip_path, args.device)
    net_lpips = None
    if args.lambda_lpips:
        from src.my_utils.utils import load_lpips
        net_lpips = load_lpips(args.vgg_path, args.lpips_path, device)
    net_disc, optimizer_disc = None, None
    if args.lambda_gan:
        from src.vision_aided_loss.cv_discriminator import CLIPDiscriminator
        net_disc = CLIPDiscriminator(args.clip_path).to(device)
        optimizer_disc = torch.optim.AdamW(net_disc.decoder.parameters(), lr=args.learning_rate,
                                          betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.adam_weight_decay,
                                          eps=args.adam_epsilon)
    scheduler_args = dict(num_warmup_steps=args.lr_warmup_steps, num_training_steps=args.max_train_steps,
                          num_cycles=args.lr_num_cycles, power=args.lr_power)
    lr_scheduler = get_scheduler(args.lr_scheduler, optimizer=optimizer, **scheduler_args)
    lr_scheduler_disc = get_scheduler(args.lr_scheduler, optimizer=optimizer_disc, **scheduler_args) if optimizer_disc else None
    scaler = torch.cuda.amp.GradScaler(enabled=args.mixed_precision == "fp16")
    start_step = 0
    if resume:
        net_proto_cls.load_state_dict(resume["prototypes"], strict=True)
        optimizer.load_state_dict(resume["optimizer"])
        for parameter, state in optimizer.state.items():
            for key in ("exp_avg", "exp_avg_sq"):
                if key in state and state[key].shape != parameter.shape:
                    raise ValueError(f"Optimizer parameter shape mismatch: {state[key].shape} vs {parameter.shape}")
        if net_disc:
            net_disc.decoder.load_state_dict(resume["discriminator"], strict=True)
            optimizer_disc.load_state_dict(resume["disc_optimizer"])
        scaler.load_state_dict(resume["scaler"])
        if "lr_scheduler" in resume:
            lr_scheduler.load_state_dict(resume["lr_scheduler"])
        if lr_scheduler_disc and resume.get("lr_scheduler_disc"):
            lr_scheduler_disc.load_state_dict(resume["lr_scheduler_disc"])
        start_step = resume["step"]
        random.setstate(resume["rng_python"])
        np.random.set_state(resume["rng_numpy"])
        torch.set_rng_state(resume["rng_torch"])
        torch.cuda.set_rng_state_all(resume["rng_cuda"])
        del resume
    if start_step >= args.max_train_steps:
        raise ValueError("--max-steps must exceed the resumed step")
    dataset_opt = dict(args.dataset_config["datasets"]["train"])
    dataset_opt.update(dataroot=str(args.train_root), de_type=args.tasks, patch_size=args.resolution,
                       seed=data_seed, samples=args.max_train_steps * args.train_batch_size * args.gradient_accumulation_steps,
                       limit_per_task=args.limit_per_task, synthetic_noise_jpeg=not args.paired_noise_jpeg)
    dataset_train = IRTrainDataset(dataset_opt)
    dataset = dataset_train
    counts = {task: len(pairs) for task, pairs in dataset.pairs.items()}
    (args.output_dir / "dataset_summary.json").write_text(json.dumps(counts, indent=2) + "\n")
    print("Training pairs:", counts, flush=True)
    loader_generator = torch.Generator().manual_seed(data_seed)
    loader = iter(DataLoader(dataset, batch_size=args.train_batch_size,
                            sampler=range(start_step * args.train_batch_size * args.gradient_accumulation_steps, len(dataset)),
                            num_workers=args.dataloader_num_workers, generator=loader_generator))
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device)[None, :, None, None]
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device)[None, :, None, None]
    def amp():
        return torch.autocast("cuda", dtype=torch.bfloat16 if args.mixed_precision == "bf16" else torch.float16) if args.mixed_precision != "no" else nullcontext()
    def context(image):
        with torch.no_grad(), amp():
            features = F.interpolate(image * 0.5 + 0.5, (224, 224), mode="bilinear", align_corners=True)
            image_context, degradation_context, _ = net_daclip.encode_image((features - mean) / std, control=True)
        return image_context.float(), degradation_context.float()
    def update(opt, params):
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(params, args.max_grad_norm, error_if_nonfinite=True)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
    history = args.output_dir / "training.jsonl"
    for step in range(start_step + 1, args.max_train_steps + 1):
        batches = []
        for _ in range(args.gradient_accumulation_steps):
            batch = next(loader)
            image_context, degradation_context = context(batch["degrad_patch"].to(device))
            batches.append((batch, image_context, degradation_context))
        stats, fake_batches = {}, []
        net_sr.train()
        net_sr.text_encoder.eval()
        for phase in ("deg", "vq"):
            net_sr.set_lora(classify=(phase == "deg"))
            if net_disc:
                net_disc.decoder.requires_grad_(False)
            optimizer.zero_grad(set_to_none=True)
            for batch, image_context, degradation_context in batches:
                image, target, labels = batch["degrad_patch"].to(device), batch["clean_patch"].to(device), batch["de_label"].to(device)
                with amp():
                    if phase == "deg":
                        prediction, vae_select, unet_select = net_sr(image, image_context, degradation_context,
                                                                   [args.pos_prompt] * len(image), classify=True)
                    else:
                        prediction = net_sr(image, image_context, degradation_context, [args.pos_prompt] * len(image))
                        vae_select = unet_select = None
                    if phase == "deg":
                        vae_cls, unet_cls, _, _ = net_proto_cls(vae_select, unet_select)
                        terms = {"l1": F.l1_loss(prediction.float(), target.float()) * args.lambda_l1,
                                 "cls": (soft_target_ce_loss(vae_cls.float(), labels) + soft_target_ce_loss(unet_cls.float(), labels)) * args.lambda_cls}
                        if args.lambda_lpips:
                            terms["lpips"] = net_lpips(prediction.float(), target.float()).mean() * args.lambda_lpips
                        if args.lambda_lpips:
                            terms["ssim"] = ssim_loss(prediction.float(), target.float()) * args.lambda_lpips
                        if args.use_contrastive_loss and args.lambda_contra:
                            terms["contra"] = (contrastive_select_loss(vae_select.float(), len(TASKS)) + contrastive_select_loss(unet_select.float(), len(TASKS))) * args.lambda_contra
                    else:
                        terms = {"vq_l1": F.l1_loss(prediction.float(), target.float()) * args.lambda_l1}
                        if net_disc:
                            terms["gan_g"] = net_disc(prediction, for_G=True) * args.lambda_gan
                        fake_batches.append(prediction.detach().cpu())
                    loss = sum(terms.values()) / args.gradient_accumulation_steps
                if not torch.isfinite(loss):
                    raise ValueError(f"Non-finite loss at step {step}, phase {phase}")
                scaler.scale(loss).backward()
                for name, value in terms.items():
                    stats[name] = stats.get(name, 0.0) + float(value.detach()) / args.gradient_accumulation_steps
                del loss, terms, prediction, vae_select, unet_select
            check_gradients(net_sr, phase)
            update(optimizer, trainable)
        if net_disc:
            net_disc.decoder.requires_grad_(True)
            for real in (True, False):
                optimizer_disc.zero_grad(set_to_none=True)
                for i, (batch, _, _) in enumerate(batches):
                    with amp():
                        image = batch["clean_patch"].to(device) if real else fake_batches[i].to(device)
                        loss = net_disc(image, for_real=real) * args.lambda_gan / args.gradient_accumulation_steps
                    scaler.scale(loss).backward()
                    name = "gan_d_real" if real else "gan_d_fake"
                    stats[name] = stats.get(name, 0.0) + float(loss.detach())
                update(optimizer_disc, list(net_disc.decoder.parameters()))
        lr_scheduler.step()
        if lr_scheduler_disc:
            lr_scheduler_disc.step()
        row = {"step": step, **stats, "tasks": [task for batch, _, _ in batches for task in batch["degra"]]}
        with history.open("a") as stream:
            stream.write(json.dumps(row) + "\n")
        print(f"step={step} " + " ".join(f"{k}={v:.5f}" for k, v in stats.items()), flush=True)
        if args.eval_freq and (step - 1) % args.eval_freq == 0:
            validate(net_sr, context, args, step, amp, device)
        if step == args.max_train_steps or (args.checkpointing_steps and (step - 1) % args.checkpointing_steps == 0):
            state = net_sr.checkpoint_state()
            torch.save(state, checkpoints / f"model_{step}.pkl")
            net_sr.export_deg(checkpoints / f"deg_model_{step}.pkl")
            net_proto_cls.save_model(checkpoints / f"proto_cls_{step}.pkl")
            if net_disc:
                torch.save(net_disc.decoder.state_dict(), checkpoints / f"disc_{step}.pkl")
            training_state = {"step": step, "model": state, "prototypes": net_proto_cls.state_dict(),
                              "optimizer": optimizer.state_dict(), "discriminator": net_disc.decoder.state_dict() if net_disc else None,
                              "disc_optimizer": optimizer_disc.state_dict() if net_disc else None, "scaler": scaler.state_dict(),
                              "lr_scheduler": lr_scheduler.state_dict(),
                              "lr_scheduler_disc": lr_scheduler_disc.state_dict() if lr_scheduler_disc else None,
                              "settings": settings, "rng_python": random.getstate(), "rng_numpy": np.random.get_state(),
                              "rng_torch": torch.get_rng_state(), "rng_cuda": torch.cuda.get_rng_state_all()}
            temp = checkpoints / "latest_resume.tmp"
            torch.save(training_state, temp)
            temp.replace(checkpoints / "latest_resume.pt")
            del state, training_state
            print(f"Saved full deg+vq, deg-only and resumable checkpoints at step {step}", flush=True)
        del batches, fake_batches
        gc.collect()
    print("Training completed", flush=True)


@torch.no_grad()
def validate(net_sr, context, args, step, amp, device):
    folder = args.output_dir / "validation" / f"step_{step}"
    folder.mkdir(parents=True, exist_ok=True)
    # Isolate validation RNG consumption from the subsequent training stream.
    rng_cpu, rng_cuda = torch.get_rng_state(), torch.cuda.get_rng_state_all()
    net_sr.eval()
    net_sr.set_lora(classify=(args.validation_mode == "deg"))
    rows = []
    try:
        if args.seed is not None:
            torch.manual_seed(args.seed)
        samples = discover(None, args.val_root, folder, args.tasks, 1)
        for sample in samples:
            image, size = load_input(sample.source)
            gt, _ = load_input(sample.reference)
            h, w = image.shape[-2:]
            crop = min(args.resolution, h, w)
            y, x = (h - crop) // 2, (w - crop) // 2
            image, gt = image[..., y:y + crop, x:x + crop].to(device), gt[..., y:y + crop, x:x + crop].to(device)
            with amp():
                image_context, degradation_context = context(image)
                pred = net_sr(image, image_context, degradation_context, [args.pos_prompt]).float()
            mse = F.mse_loss(pred * 0.5 + 0.5, gt * 0.5 + 0.5).item()
            psnr = -10 * np.log10(max(mse, 1e-12))
            rows.append({"task": sample.task, "psnr": float(psnr), "crop_size": crop})
            array = pred[0].add(1).mul(127.5).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
            Image.fromarray(array).save(folder / f"{sample.task}.png")
        (folder / "metrics.json").write_text(json.dumps({"protocol": "center patch, float RGB PSNR",
                                                        "adapter_mode": args.validation_mode, "images": rows}, indent=2) + "\n")
    finally:
        torch.set_rng_state(rng_cpu)
        torch.cuda.set_rng_state_all(rng_cuda)
        net_sr.train()
        net_sr.text_encoder.eval()


if __name__ == "__main__":
    args = parse_args_paired_training()
    main(args)
