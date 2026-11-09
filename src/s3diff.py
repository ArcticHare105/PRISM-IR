import torch
from torch import nn
from transformers import AutoTokenizer, CLIPTextModel
from diffusers import AutoencoderKL, UNet2DConditionModel
from peft import LoraConfig
from .model import (
    make_1step_sched,
    my_lora_fwd_simple,
    my_vae_encoder_fwd,
    my_vae_decoder_fwd,
)
from src.my_utils.utils import load_restoration_state


def Round_Sigmoid_Gumbel(logits, tau=1, hard=False, deterministic=False):
    if deterministic:
        if hard:
            return (logits >= 0).to(logits.dtype)
        return torch.sigmoid(logits / tau)

    gumbels = -torch.empty_like(logits).exponential_().log()
    gumbels = logits + gumbels

    y_soft = torch.sigmoid(gumbels / tau)

    if hard:
        y_hard = torch.round(y_soft)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        ret = y_soft

    return ret


class S3Diff(torch.nn.Module):
    def __init__(
        self,
        sd_path=None,
        pretrained_path=None,
        lora_rank_unet_deg=None,
        lora_rank_vae_deg=None,
        lora_rank_unet_vq=None,
        lora_rank_vae_vq=None,
        device="cuda",
        load_vq=True,
        timestep=300,
        denoising_mode="scheduler",
    ):
        super().__init__()
        if denoising_mode not in ("scheduler", "prediction_minus_latent"):
            raise ValueError(f"Unknown denoising_mode: {denoising_mode}")
        self.denoising_mode = denoising_mode
        sd = (
            torch.load(pretrained_path, map_location="cpu")
            if pretrained_path is not None
            else None
        )

        def checkpoint_rank(name, requested, fallback):
            if requested is not None:
                return requested
            return sd.get(name, fallback) if sd is not None else fallback

        lora_rank_unet_deg = checkpoint_rank("rank_unet_deg", lora_rank_unet_deg, 64)
        lora_rank_vae_deg = checkpoint_rank("rank_vae_deg", lora_rank_vae_deg, 48)
        lora_rank_unet_vq = checkpoint_rank("rank_unet_vq", lora_rank_unet_vq, 8)
        lora_rank_vae_vq = checkpoint_rank("rank_vae_vq", lora_rank_vae_vq, 4)
        self.load_vq = load_vq
        if sd is not None and not load_vq:
            for field in ("state_dict_unet", "state_dict_vae"):
                sd[field] = {
                    k: v
                    for k, v in sd[field].items()
                    if ".unet_vq." not in k and ".vae_vq." not in k
                }
        if sd is not None:
            requested = {
                "rank_unet_deg": lora_rank_unet_deg,
                "rank_vae_deg": lora_rank_vae_deg,
                "rank_unet_vq": lora_rank_unet_vq,
                "rank_vae_vq": lora_rank_vae_vq,
            }
            for name, rank in requested.items():
                if name.endswith("_vq") and not load_vq:
                    continue
                if name not in sd:
                    raise ValueError(
                        f"Checkpoint is missing {name}; deg+vq inference requires a full restoration checkpoint"
                    )
                if rank != sd[name]:
                    raise ValueError(
                        f"{name}: requested {rank}, checkpoint has {sd[name]}"
                    )
        self.tokenizer = AutoTokenizer.from_pretrained(
            sd_path, subfolder="tokenizer", local_files_only=True
        )
        self.text_encoder = CLIPTextModel.from_pretrained(
            sd_path, subfolder="text_encoder", local_files_only=True
        ).to(device)
        self.sched = make_1step_sched(sd_path, device)
        self.tau = 1.0

        vae = AutoencoderKL.from_pretrained(
            sd_path, subfolder="vae", local_files_only=True
        )
        unet = UNet2DConditionModel.from_pretrained(
            sd_path, subfolder="unet", local_files_only=True
        )

        vae.encoder.forward = my_vae_encoder_fwd.__get__(
            vae.encoder, vae.encoder.__class__
        )
        vae.decoder.forward = my_vae_decoder_fwd.__get__(
            vae.decoder, vae.decoder.__class__
        )
        vae.decoder.skip_conv_1 = torch.nn.Conv2d(
            512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False
        ).to(device)
        vae.decoder.skip_conv_2 = torch.nn.Conv2d(
            256, 512, kernel_size=(1, 1), stride=(1, 1), bias=False
        ).to(device)
        vae.decoder.skip_conv_3 = torch.nn.Conv2d(
            128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False
        ).to(device)
        vae.decoder.skip_conv_4 = torch.nn.Conv2d(
            128, 256, kernel_size=(1, 1), stride=(1, 1), bias=False
        ).to(device)
        vae.decoder.ignore_skip = False

        target_modules_vae = [
            "conv1",
            "conv2",
            "conv_in",
            "conv_shortcut",
            "conv",
            "conv_out",
            "to_k",
            "to_q",
            "to_v",
            "to_out.0",
        ]
        target_modules_unet = [
            "to_k",
            "to_q",
            "to_v",
            "to_out.0",
            "conv",
            "conv1",
            "conv2",
            "conv_shortcut",
            "conv_out",
            "proj_in",
            "proj_out",
            "ff.net.2",
            "ff.net.0.proj",
        ]

        self.vae_selector = nn.Sequential(
            nn.Linear(1024, lora_rank_vae_deg * 4),
            nn.GELU(),
            nn.Linear(lora_rank_vae_deg * 4, lora_rank_vae_deg * 2),
            nn.GELU(),
            nn.Linear(lora_rank_vae_deg * 2, lora_rank_vae_deg),
        )

        self.unet_selector = nn.Sequential(
            nn.Linear(1024, lora_rank_unet_deg * 4),
            nn.GELU(),
            nn.Linear(lora_rank_unet_deg * 4, lora_rank_unet_deg * 2),
            nn.GELU(),
            nn.Linear(lora_rank_unet_deg * 2, lora_rank_unet_deg),
        )
        self._linear_initialize_weights()

        if pretrained_path is not None:
            vae_deg_lora_config = LoraConfig(
                r=sd["rank_vae_deg"],
                init_lora_weights="gaussian",
                target_modules=sd["vae_lora_target_modules"],
            )
            vae.add_adapter(vae_deg_lora_config, adapter_name="vae_deg")
            if load_vq:
                vae_vq_lora_config = LoraConfig(
                    r=sd["rank_vae_vq"],
                    init_lora_weights="gaussian",
                    target_modules=sd["vae_lora_target_modules"],
                )
                vae.add_adapter(vae_vq_lora_config, adapter_name="vae_vq")
            vae.set_adapter(["vae_deg", "vae_vq"] if load_vq else ["vae_deg"])
            load_restoration_state(
                vae, sd["state_dict_vae"], lambda k: "lora" in k or "skip_conv" in k
            )

            unet_deg_lora_config = LoraConfig(
                r=sd["rank_unet_deg"],
                init_lora_weights="gaussian",
                target_modules=sd["unet_lora_target_modules"],
            )
            unet.add_adapter(unet_deg_lora_config, adapter_name="unet_deg")
            if load_vq:
                unet_vq_lora_config = LoraConfig(
                    r=sd["rank_unet_vq"],
                    init_lora_weights="gaussian",
                    target_modules=sd["unet_lora_target_modules"],
                )
                unet.add_adapter(unet_vq_lora_config, adapter_name="unet_vq")
            unet.set_adapter(["unet_deg", "unet_vq"] if load_vq else ["unet_deg"])
            load_restoration_state(
                unet,
                sd["state_dict_unet"],
                lambda k: "lora" in k or k.startswith("conv_in."),
            )

            self.vae_selector.load_state_dict(
                sd["state_dict_vae_selector"], strict=True
            )
            self.unet_selector.load_state_dict(
                sd["state_dict_unet_selector"], strict=True
            )
        else:
            print(
                "Loaded SD-Turbo; initializing fresh deg/vq LoRA, selectors and skip convolutions",
                flush=True,
            )
            torch.nn.init.constant_(vae.decoder.skip_conv_1.weight, 1e-5)
            torch.nn.init.constant_(vae.decoder.skip_conv_2.weight, 1e-5)
            torch.nn.init.constant_(vae.decoder.skip_conv_3.weight, 1e-5)
            torch.nn.init.constant_(vae.decoder.skip_conv_4.weight, 1e-5)

            vae_deg_lora_config = LoraConfig(
                r=lora_rank_vae_deg,
                init_lora_weights="gaussian",
                target_modules=target_modules_vae,
            )
            vae.add_adapter(vae_deg_lora_config, adapter_name="vae_deg")
            if load_vq:
                vae_vq_lora_config = LoraConfig(
                    r=lora_rank_vae_vq,
                    init_lora_weights="gaussian",
                    target_modules=target_modules_vae,
                )
                vae.add_adapter(vae_vq_lora_config, adapter_name="vae_vq")
            vae.set_adapter(["vae_deg", "vae_vq"] if load_vq else ["vae_deg"])

            unet_deg_lora_config = LoraConfig(
                r=lora_rank_unet_deg,
                init_lora_weights="gaussian",
                target_modules=target_modules_unet,
            )
            unet.add_adapter(unet_deg_lora_config, adapter_name="unet_deg")
            if load_vq:
                unet_vq_lora_config = LoraConfig(
                    r=lora_rank_unet_vq,
                    init_lora_weights="gaussian",
                    target_modules=target_modules_unet,
                )
                unet.add_adapter(unet_vq_lora_config, adapter_name="unet_vq")
            unet.set_adapter(["unet_deg", "unet_vq"] if load_vq else ["unet_deg"])

        self.lora_rank_unet_deg = lora_rank_unet_deg
        self.lora_rank_vae_deg = lora_rank_vae_deg
        self.lora_rank_unet_vq = lora_rank_unet_vq
        self.lora_rank_vae_vq = lora_rank_vae_vq

        self.target_modules_vae = (
            sd["vae_lora_target_modules"] if sd is not None else target_modules_vae
        )
        self.target_modules_unet = (
            sd["unet_lora_target_modules"] if sd is not None else target_modules_unet
        )

        self.vae_lora_layers = []
        for name, module in vae.named_modules():
            if "base_layer" in name:
                self.vae_lora_layers.append(name[: -len(".base_layer")])

        for name, module in vae.named_modules():
            if name in self.vae_lora_layers:
                module.forward = my_lora_fwd_simple.__get__(module, module.__class__)

        self.unet_lora_layers = []
        for name, module in unet.named_modules():
            if "base_layer" in name:
                self.unet_lora_layers.append(name[: -len(".base_layer")])

        for name, module in unet.named_modules():
            if name in self.unet_lora_layers:
                module.forward = my_lora_fwd_simple.__get__(module, module.__class__)

        unet.to(device)
        vae.to(device)
        self.unet, self.vae = unet, vae
        self.vae.decoder.gamma = 1
        self.register_buffer(
            "timesteps",
            torch.tensor([timestep], device=device, dtype=torch.long),
            persistent=False,
        )
        self.vae_selector.to(device)
        self.unet_selector.to(device)
        self.rank_vae = lora_rank_vae_deg
        self.rank_unet = lora_rank_unet_deg
        self.text_encoder.requires_grad_(False)

    def _linear_initialize_weights(self):
        for modules in [self.vae_selector, self.unet_selector]:
            for module in modules:
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

    def set_eval(self):
        self.eval()
        self.unet.eval()
        self.vae.eval()
        self.vae_selector.eval()
        self.unet_selector.eval()
        self.unet.requires_grad_(False)
        self.vae.requires_grad_(False)

    def set_train(self):
        self.train()

        self.unet.requires_grad_(False)
        self.vae.requires_grad_(False)
        self.vae_selector.requires_grad_(True)
        self.unet_selector.requires_grad_(True)
        self.text_encoder.eval()
        self.unet.train()
        self.vae.train()
        self.vae_selector.train()
        self.unet_selector.train()

        for n, _p in self.unet.named_parameters():
            if "lora" in n:
                _p.requires_grad = True
        self.unet.conv_in.requires_grad_(True)

        for n, _p in self.vae.named_parameters():
            if "lora" in n:
                _p.requires_grad = True

        self.vae.decoder.skip_conv_1.requires_grad_(True)
        self.vae.decoder.skip_conv_2.requires_grad_(True)
        self.vae.decoder.skip_conv_3.requires_grad_(True)
        self.vae.decoder.skip_conv_4.requires_grad_(True)

    def set_lora(self, classify):
        if not classify and not self.load_vq:
            raise ValueError("VQ is disabled in the deg-only test model")
        if classify:
            self.vae.set_adapter(["vae_deg"])
            self.unet.set_adapter(["unet_deg"])
            self.change_weights_requires_grad(
                self.vae, requires_grad=True, pattern_match_name="vae_deg"
            )
            self.change_weights_requires_grad(
                self.unet, requires_grad=True, pattern_match_name="unet_deg"
            )
        else:
            self.vae.set_adapter(["vae_deg", "vae_vq"])
            self.unet.set_adapter(["unet_deg", "unet_vq"])
            self.change_weights_requires_grad(
                self.vae, requires_grad=False, pattern_match_name="vae_deg"
            )
            self.change_weights_requires_grad(
                self.unet, requires_grad=False, pattern_match_name="unet_deg"
            )

    def change_weights_requires_grad(self, model, requires_grad, pattern_match_name):
        for name, parameter in model.named_parameters():
            if pattern_match_name in name:
                parameter.requires_grad_(requires_grad)

    def forward(self, c_t, img_c, deg_c, prompt, classify=False):
        if prompt is None:
            prompt = [""] * len(c_t)
        caption_tokens = self.tokenizer(
            prompt,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(c_t.device)
        caption_enc = self.text_encoder(caption_tokens)[0]

        vae_logits = self.vae_selector(torch.cat([deg_c, img_c], -1))
        unet_logits = self.unet_selector(torch.cat([deg_c, img_c], -1))

        deterministic_gate = not self.training
        vae_select = Round_Sigmoid_Gumbel(
            vae_logits, self.tau, True, deterministic=deterministic_gate
        )
        unet_select = Round_Sigmoid_Gumbel(
            unet_logits, self.tau, True, deterministic=deterministic_gate
        )

        vae_select_mat = torch.diag_embed(vae_select)
        unet_select_mat = torch.diag_embed(unet_select)

        for layer_name, module in self.vae.named_modules():
            if layer_name in self.vae_lora_layers:
                module.de_mod = vae_select_mat

        for layer_name, module in self.unet.named_modules():
            if layer_name in self.unet_lora_layers:
                module.de_mod = unet_select_mat

        encoded_control = (
            self.vae.encode(c_t).latent_dist.sample() * self.vae.config.scaling_factor
        )
        model_pred = self.unet(
            encoded_control,
            self.timesteps,
            encoder_hidden_states=caption_enc,
        ).sample
        if self.denoising_mode == "prediction_minus_latent":
            x_denoised = model_pred - encoded_control
        else:
            x_denoised = self.sched.step(
                model_pred, self.timesteps, encoded_control, return_dict=True
            ).prev_sample
        self.vae.decoder.incoming_skip_acts = self.vae.encoder.current_down_blocks
        output_image = (
            self.vae.decode(x_denoised / self.vae.config.scaling_factor).sample
        ).clamp(-1, 1)

        if classify:
            return output_image, torch.sigmoid(vae_logits), torch.sigmoid(unet_logits)
        else:
            return output_image

    def checkpoint_state(self):
        sd = {}
        sd["unet_lora_target_modules"] = self.target_modules_unet
        sd["vae_lora_target_modules"] = self.target_modules_vae
        sd["rank_unet_deg"] = self.lora_rank_unet_deg
        sd["rank_vae_deg"] = self.lora_rank_vae_deg
        sd["rank_unet_vq"] = self.lora_rank_unet_vq
        sd["rank_vae_vq"] = self.lora_rank_vae_vq

        sd["state_dict_unet"] = {
            k: v
            for k, v in self.unet.state_dict().items()
            if "lora" in k or "conv_in" in k
        }
        sd["state_dict_vae"] = {
            k: v
            for k, v in self.vae.state_dict().items()
            if "lora" in k or "skip_conv" in k
        }

        sd["state_dict_vae_selector"] = {
            k: v for k, v in self.vae_selector.state_dict().items()
        }
        sd["state_dict_unet_selector"] = {
            k: v for k, v in self.unet_selector.state_dict().items()
        }
        for field in (
            "state_dict_unet",
            "state_dict_vae",
            "state_dict_vae_selector",
            "state_dict_unet_selector",
        ):
            sd[field] = {k: v.detach().cpu().clone() for k, v in sd[field].items()}
        return sd

    def save_model(self, outf):
        torch.save(self.checkpoint_state(), outf)

    def set_phase(self, phase):
        if phase not in ("deg", "vq"):
            raise ValueError(phase)
        self.set_lora(classify=(phase == "deg"))

    def export_deg(self, outf):
        sd = self.checkpoint_state()
        sd.pop("rank_unet_vq")
        sd.pop("rank_vae_vq")
        for field in ("state_dict_unet", "state_dict_vae"):
            sd[field] = {
                k: v
                for k, v in sd[field].items()
                if ".vae_vq." not in k and ".unet_vq." not in k
            }
        torch.save(sd, outf)
