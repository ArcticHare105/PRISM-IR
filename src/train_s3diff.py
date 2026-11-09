#!/usr/bin/env python3

from contextlib import nullcontext
import gc
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from diffusers.optimization import get_scheduler

from src.my_utils.data_utils import discover, load_input
from src.my_utils.utils import load_daclip, resolve_sd_path, set_seed
from src.my_utils.model_checks import (
    check_checkpoint,
    check_fresh_initialization,
    check_gradients,
)
from src.my_utils.dataset_utils import IRTrainDataset
from src.model import contrastive_select_loss, soft_target_ce_loss, ssim_loss
from src.s3diff import S3Diff
from src.classify_model import proto_DegraCls

from src.my_utils.training_utils import parse_args_paired_training


def check_resume_settings(saved_settings, settings):
    saved_settings.setdefault("inference_timestep", 300)
    saved_settings.setdefault("denoising_mode", "scheduler")
    aliases = {
        "train_batch_size": "batch_size",
        "gradient_accumulation_steps": "gradient_accumulation",
        "resolution": "patch_size",
        "mixed_precision": "precision",
    }
    keys = (
        "tasks",
        "seed",
        "train_batch_size",
        "gradient_accumulation_steps",
        "resolution",
        "paired_noise_jpeg",
        "mixed_precision",
        "inference_timestep",
        "denoising_mode",
        "learning_rate",
        "lr_scheduler",
        "lr_warmup_steps",
        "lr_num_cycles",
        "lr_power",
        "lambda_l1",
        "lambda_lpips",
        "lambda_cls",
        "lambda_contra",
        "use_contrastive_loss",
        "lambda_gan",
        "train_root",
    )
    for key in keys:
        saved = saved_settings.get(key, saved_settings.get(aliases.get(key)))
        if key == "mixed_precision" and saved == "fp32":
            saved = "no"
        if saved != settings[key]:
            raise ValueError(f"Resume setting differs: {key}")


def degradation_losses(
    prediction, target, labels, vae_select, unet_select, classifier, lpips, args
):
    vae_cls, unet_cls, _, _ = classifier(vae_select, unet_select)
    losses = {
        "l1": F.l1_loss(prediction.float(), target.float()) * args.lambda_l1,
        "cls": (
            soft_target_ce_loss(vae_cls.float(), labels)
            + soft_target_ce_loss(unet_cls.float(), labels)
        )
        * args.lambda_cls,
    }
    if args.lambda_lpips:
        losses["lpips"] = (
            lpips(prediction.float(), target.float()).mean() * args.lambda_lpips
        )
        losses["ssim"] = (
            ssim_loss(prediction.float(), target.float()) * args.lambda_lpips
        )
    if args.use_contrastive_loss and args.lambda_contra:
        losses["contra"] = (
            contrastive_select_loss(vae_select.float(), len(args.tasks))
            + contrastive_select_loss(unet_select.float(), len(args.tasks))
        ) * args.lambda_contra
    return losses


def save_checkpoint(
    directory,
    step,
    model,
    classifier,
    discriminator,
    optimizer,
    disc_optimizer,
    scaler,
    scheduler,
    disc_scheduler,
    settings,
):
    model_state = model.checkpoint_state()
    torch.save(model_state, directory / f"model_{step}.pkl")
    model.export_deg(directory / f"deg_model_{step}.pkl")
    classifier.save_model(directory / f"proto_cls_{step}.pkl")
    if discriminator:
        torch.save(discriminator.decoder.state_dict(), directory / f"disc_{step}.pkl")
    training_state = {
        "step": step,
        "model": model_state,
        "prototypes": classifier.state_dict(),
        "optimizer": optimizer.state_dict(),
        "discriminator": discriminator.decoder.state_dict() if discriminator else None,
        "disc_optimizer": disc_optimizer.state_dict() if disc_optimizer else None,
        "scaler": scaler.state_dict(),
        "lr_scheduler": scheduler.state_dict(),
        "lr_scheduler_disc": disc_scheduler.state_dict() if disc_scheduler else None,
        "settings": settings,
    }
    temporary = directory / "latest_resume.tmp"
    torch.save(training_state, temporary)
    temporary.replace(directory / "latest_resume.pt")


def main(args):
    torch.set_num_threads(4)
    args.sd_path = resolve_sd_path(args.sd_path)
    device = torch.device(args.device)
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints = args.output_dir / "checkpoints"
    checkpoints.mkdir(exist_ok=True)
    settings = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    resume = torch.load(args.resume, map_location="cpu") if args.resume else None
    initial_checkpoint = None
    if resume:
        initial_checkpoint = args.output_dir / "resume_initial_model.pkl"
        torch.save(resume["model"], initial_checkpoint)
        check_resume_settings(resume["settings"], settings)
    restoration_model = S3Diff(
        sd_path=str(args.sd_path),
        pretrained_path=initial_checkpoint,
        lora_rank_unet_deg=args.lora_rank_unet_deg,
        lora_rank_vae_deg=args.lora_rank_vae_deg,
        lora_rank_unet_vq=args.lora_rank_unet_vq,
        lora_rank_vae_vq=args.lora_rank_vae_vq,
        device=args.device,
        timestep=args.inference_timestep,
        denoising_mode=args.denoising_mode,
    )
    restoration_model.set_train()
    structure = (
        check_checkpoint(restoration_model, initial_checkpoint)
        if resume
        else check_fresh_initialization(restoration_model)
    )
    for network in ("unet", "vae"):
        for adapter in ("deg", "vq"):
            name = f"lora_rank_{network}_{adapter}"
            setattr(args, name, getattr(restoration_model, name))
            settings[name] = getattr(restoration_model, name)
    settings["model"] = structure
    settings["initial_checkpoint"] = str(args.resume) if args.resume else None
    settings["base_model"] = str(args.sd_path)
    settings["training_mode"] = "resume" if args.resume else "fresh_lora_on_sd_turbo"
    (args.output_dir / "settings.json").write_text(
        json.dumps(settings, indent=2) + "\n"
    )
    (args.output_dir / "model_structure.json").write_text(
        json.dumps(structure, indent=2) + "\n"
    )
    if resume:
        initial_checkpoint.unlink()
    print(
        "Resume weights and phases verified:"
        if resume
        else "Fresh initialization and phases verified:",
        structure["ranks"],
        flush=True,
    )
    if args.check_only:
        return
    if args.gradient_checkpointing:
        restoration_model.unet.enable_gradient_checkpointing()
    if args.enable_xformers_memory_efficient_attention:
        restoration_model.unet.enable_xformers_memory_efficient_attention()
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
    classifier = proto_DegraCls(
        restoration_model.rank_vae,
        restoration_model.rank_unet,
        class_num=len(args.tasks),
    ).to(device)

    trainable = [
        p
        for module in (
            restoration_model.vae,
            restoration_model.unet,
            restoration_model.vae_selector,
            restoration_model.unet_selector,
        )
        for p in module.parameters()
        if p.requires_grad
    ] + list(classifier.parameters())
    optimizer = torch.optim.AdamW(
        trainable,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    daclip = load_daclip(args.daclip_path, args.device)
    lpips_model = None
    if args.lambda_lpips:
        from src.my_utils.utils import load_lpips

        lpips_model = load_lpips(args.vgg_path, args.lpips_path, device)
    discriminator, discriminator_optimizer = None, None
    if args.lambda_gan:
        from src.vision_aided_loss.cv_discriminator import CLIPDiscriminator

        discriminator = CLIPDiscriminator(args.clip_path).to(device)
        discriminator_optimizer = torch.optim.AdamW(
            discriminator.decoder.parameters(),
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )
    scheduler_args = dict(
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=args.max_train_steps,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )
    lr_scheduler = get_scheduler(
        args.lr_scheduler, optimizer=optimizer, **scheduler_args
    )
    discriminator_scheduler = (
        get_scheduler(
            args.lr_scheduler, optimizer=discriminator_optimizer, **scheduler_args
        )
        if discriminator_optimizer
        else None
    )
    scaler = torch.cuda.amp.GradScaler(enabled=args.mixed_precision == "fp16")
    start_step = 0
    if resume:
        classifier.load_state_dict(resume["prototypes"], strict=True)
        optimizer.load_state_dict(resume["optimizer"])
        for parameter, state in optimizer.state.items():
            for key in ("exp_avg", "exp_avg_sq"):
                if key in state and state[key].shape != parameter.shape:
                    raise ValueError(
                        f"Optimizer parameter shape mismatch: {state[key].shape} vs {parameter.shape}"
                    )
        if discriminator:
            discriminator.decoder.load_state_dict(resume["discriminator"], strict=True)
            discriminator_optimizer.load_state_dict(resume["disc_optimizer"])
        scaler.load_state_dict(resume["scaler"])
        if "lr_scheduler" in resume:
            lr_scheduler.load_state_dict(resume["lr_scheduler"])
        if discriminator_scheduler and resume.get("lr_scheduler_disc"):
            discriminator_scheduler.load_state_dict(resume["lr_scheduler_disc"])
        start_step = resume["step"]
        del resume
    if start_step >= args.max_train_steps:
        raise ValueError("--max-steps must exceed the resumed step")
    dataset_opt = dict(args.dataset_config["datasets"]["train"])
    dataset_opt.update(
        dataroot=str(args.train_root),
        de_type=args.tasks,
        patch_size=args.resolution,
        samples=args.max_train_steps
        * args.train_batch_size
        * args.gradient_accumulation_steps,
        limit_per_task=args.limit_per_task,
        synthetic_noise_jpeg=not args.paired_noise_jpeg,
    )
    dataset = IRTrainDataset(dataset_opt)
    counts = {task: len(pairs) for task, pairs in dataset.pairs.items()}
    (args.output_dir / "dataset_summary.json").write_text(
        json.dumps(counts, indent=2) + "\n"
    )
    print("Training pairs:", counts, flush=True)
    loader = iter(
        DataLoader(
            dataset,
            batch_size=args.train_batch_size,
            sampler=range(
                start_step * args.train_batch_size * args.gradient_accumulation_steps,
                len(dataset),
            ),
            num_workers=args.dataloader_num_workers,
        )
    )
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device)[
        None, :, None, None
    ]
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device)[
        None, :, None, None
    ]

    def amp():
        return (
            torch.autocast(
                "cuda",
                dtype=torch.bfloat16
                if args.mixed_precision == "bf16"
                else torch.float16,
            )
            if args.mixed_precision != "no"
            else nullcontext()
        )

    def context(image):
        with torch.no_grad(), amp():
            features = F.interpolate(
                image * 0.5 + 0.5, (224, 224), mode="bilinear", align_corners=True
            )
            image_context, degradation_context, _ = daclip.encode_image(
                (features - mean) / std, control=True
            )
        return image_context.float(), degradation_context.float()

    def update(opt, params):
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(
            params, args.max_grad_norm, error_if_nonfinite=True
        )
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)

    history = args.output_dir / "training.jsonl"
    for step in range(start_step + 1, args.max_train_steps + 1):
        batches = []
        for _ in range(args.gradient_accumulation_steps):
            batch = next(loader)
            image_context, degradation_context = context(
                batch["degrad_patch"].to(device)
            )
            batches.append((batch, image_context, degradation_context))
        stats, fake_batches = {}, []
        restoration_model.train()
        restoration_model.text_encoder.eval()
        for phase in ("deg", "vq"):
            restoration_model.set_lora(classify=(phase == "deg"))
            if discriminator:
                discriminator.decoder.requires_grad_(False)
            optimizer.zero_grad(set_to_none=True)
            for batch, image_context, degradation_context in batches:
                image, target, labels = (
                    batch["degrad_patch"].to(device),
                    batch["clean_patch"].to(device),
                    batch["de_label"].to(device),
                )
                with amp():
                    if phase == "deg":
                        prediction, vae_select, unet_select = restoration_model(
                            image,
                            image_context,
                            degradation_context,
                            [args.pos_prompt] * len(image),
                            classify=True,
                        )
                    else:
                        prediction = restoration_model(
                            image,
                            image_context,
                            degradation_context,
                            [args.pos_prompt] * len(image),
                        )
                        vae_select = unet_select = None
                    if phase == "deg":
                        terms = degradation_losses(
                            prediction,
                            target,
                            labels,
                            vae_select,
                            unet_select,
                            classifier,
                            lpips_model,
                            args,
                        )
                    else:
                        terms = {
                            "vq_l1": F.l1_loss(prediction.float(), target.float())
                            * args.lambda_l1
                        }
                        if discriminator:
                            terms["gan_g"] = (
                                discriminator(prediction, for_G=True) * args.lambda_gan
                            )
                        fake_batches.append(prediction.detach().cpu())
                    loss = sum(terms.values()) / args.gradient_accumulation_steps
                if not torch.isfinite(loss):
                    raise ValueError(f"Non-finite loss at step {step}, phase {phase}")
                scaler.scale(loss).backward()
                for name, value in terms.items():
                    stats[name] = (
                        stats.get(name, 0.0)
                        + float(value.detach()) / args.gradient_accumulation_steps
                    )
                del loss, terms, prediction, vae_select, unet_select
            check_gradients(restoration_model, phase)
            update(optimizer, trainable)
        if discriminator:
            discriminator.decoder.requires_grad_(True)
            for real in (True, False):
                discriminator_optimizer.zero_grad(set_to_none=True)
                for i, (batch, _, _) in enumerate(batches):
                    with amp():
                        image = (
                            batch["clean_patch"].to(device)
                            if real
                            else fake_batches[i].to(device)
                        )
                        loss = (
                            discriminator(image, for_real=real)
                            * args.lambda_gan
                            / args.gradient_accumulation_steps
                        )
                    scaler.scale(loss).backward()
                    name = "gan_d_real" if real else "gan_d_fake"
                    stats[name] = stats.get(name, 0.0) + float(loss.detach())
                update(
                    discriminator_optimizer, list(discriminator.decoder.parameters())
                )
        lr_scheduler.step()
        if discriminator_scheduler:
            discriminator_scheduler.step()
        row = {
            "step": step,
            **stats,
            "tasks": [task for batch, _, _ in batches for task in batch["degra"]],
        }
        with history.open("a") as stream:
            stream.write(json.dumps(row) + "\n")
        print(
            f"step={step} " + " ".join(f"{k}={v:.5f}" for k, v in stats.items()),
            flush=True,
        )
        if args.eval_freq and (step - 1) % args.eval_freq == 0:
            validate(restoration_model, context, args, step, amp, device)
        if step == args.max_train_steps or (
            args.checkpointing_steps and (step - 1) % args.checkpointing_steps == 0
        ):
            save_checkpoint(
                checkpoints,
                step,
                restoration_model,
                classifier,
                discriminator,
                optimizer,
                discriminator_optimizer,
                scaler,
                lr_scheduler,
                discriminator_scheduler,
                settings,
            )
            print(
                f"Saved full deg+vq, deg-only and resumable checkpoints at step {step}",
                flush=True,
            )
        del batches, fake_batches
        gc.collect()
    print("Training completed", flush=True)


@torch.no_grad()
def validate(restoration_model, context, args, step, amp, device):
    folder = args.output_dir / "validation" / f"step_{step}"
    folder.mkdir(parents=True, exist_ok=True)

    restoration_model.eval()
    restoration_model.set_lora(classify=(args.validation_mode == "deg"))
    rows = []
    try:
        samples = discover(None, args.val_root, folder, args.tasks, 1)
        for sample in samples:
            image, size = load_input(sample.source)
            gt, _ = load_input(sample.reference)
            h, w = image.shape[-2:]
            crop = min(args.resolution, h, w)
            y, x = (h - crop) // 2, (w - crop) // 2
            image, gt = (
                image[..., y : y + crop, x : x + crop].to(device),
                gt[..., y : y + crop, x : x + crop].to(device),
            )
            with amp():
                image_context, degradation_context = context(image)
                pred = restoration_model(
                    image, image_context, degradation_context, [args.pos_prompt]
                ).float()
            mse = F.mse_loss(pred * 0.5 + 0.5, gt * 0.5 + 0.5).item()
            psnr = -10 * np.log10(max(mse, 1e-12))
            rows.append({"task": sample.task, "psnr": float(psnr), "crop_size": crop})
            array = (
                pred[0]
                .add(1)
                .mul(127.5)
                .clamp(0, 255)
                .byte()
                .permute(1, 2, 0)
                .cpu()
                .numpy()
            )
            Image.fromarray(array).save(folder / f"{sample.task}.png")
        (folder / "metrics.json").write_text(
            json.dumps(
                {
                    "protocol": "center patch, float RGB PSNR",
                    "adapter_mode": args.validation_mode,
                    "images": rows,
                },
                indent=2,
            )
            + "\n"
        )
    finally:
        restoration_model.train()
        restoration_model.text_encoder.eval()


if __name__ == "__main__":
    args = parse_args_paired_training()
    main(args)
