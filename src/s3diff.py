"""Full deg/vq restoration S3Diff, keeping the original model class and interfaces."""
import copy
import torch
from torch import nn
from transformers import AutoTokenizer, CLIPTextModel
from diffusers import AutoencoderKL, UNet2DConditionModel
from peft import LoraConfig
from .model import make_1step_sched, my_lora_fwd_simple, my_vae_encoder_fwd, my_vae_decoder_fwd
from src.my_utils.utils import load_restoration_state


def Round_Sigmoid_Gumbel(logits, tau=1, hard=False):
    gumbels = -torch.empty_like(logits).exponential_().log()  # Gumbel(0,1)
    gumbels = logits + gumbels  # Add Gumbel noise to logits

    # Step 2: Apply Sigmoid with temperature
    y_soft = torch.sigmoid(gumbels / tau)  # Smooth approximation of binary output

    # Step 3: If hard=True, apply hard thresholding
    if hard:
        # Hard binary output: 0 or 1
        y_hard = torch.round(y_soft)  # Force binary output
        ret = y_hard - y_soft.detach() + y_soft  # Straight-Through Estimator (STE)
    else:
        # Soft output: values in (0, 1)
        ret = y_soft

    return ret


class S3Diff(torch.nn.Module):
    def __init__(self, sd_path=None, pretrained_path=None, lora_rank_unet_deg=None, lora_rank_vae_deg=None,
        lora_rank_unet_vq=None, lora_rank_vae_vq=None, class_num=10, device="cuda", load_vq=True,
        selector_rank_unet=None, selector_rank_vae=None):
        super().__init__()
        sd = torch.load(pretrained_path, map_location="cpu") if pretrained_path is not None else None
        # Checkpoint dimensions define the model unless a rank is explicitly
        # requested. This supports both current rebased and original weights.
        def checkpoint_rank(name, requested, fallback):
            if requested is not None:
                return requested
            return sd.get(name, fallback) if sd is not None else fallback

        lora_rank_unet_deg = checkpoint_rank("rank_unet_deg", lora_rank_unet_deg, 64)
        lora_rank_vae_deg = checkpoint_rank("rank_vae_deg", lora_rank_vae_deg, 48)
        lora_rank_unet_vq = checkpoint_rank("rank_unet_vq", lora_rank_unet_vq, 8)
        lora_rank_vae_vq = checkpoint_rank("rank_vae_vq", lora_rank_vae_vq, 4)
        self.adapter_rebase = copy.deepcopy(sd.get("adapter_rebase")) if sd is not None else None
        if sd is None and (selector_rank_unet is not None or selector_rank_vae is not None):
            # Construct the tested channel layout from settings only. All LoRA
            # tensors below are initialized afresh, without a restoration file.
            if selector_rank_unet is None or selector_rank_vae is None:
                raise ValueError("Specify both selector dimensions")
            self.adapter_rebase = {
                "version": 1,
                "rule": "independently initialized deg and vq; no weight reparameterization",
                "weight_layout": "concatenated unscaled B blocks; head and tail retain original separate scales",
                "gate_rule": "dynamic selector gates followed by constant-one gates",
            }
            for network, rank, selector, vq_rank in (
                    ("unet", lora_rank_unet_deg, selector_rank_unet, lora_rank_unet_vq),
                    ("vae", lora_rank_vae_deg, selector_rank_vae, lora_rank_vae_vq)):
                if selector <= 0 or rank - selector != vq_rank or vq_rank <= 0:
                    raise ValueError(f"{network}: deg rank must equal selector dimension + vq rank")
                self.adapter_rebase.update({f"selector_rank_{network}": selector,
                                           f"fixed_rank_{network}": rank - selector,
                                           f"lora_alpha_{network}_deg": rank * 8 / selector,
                                           f"lora_alpha_{network}_vq": 8})
        if self.adapter_rebase is not None:
            rebase = self.adapter_rebase
            layout = "concatenated unscaled B blocks; head and tail retain original separate scales"
            if rebase.get("version") != 1 or rebase.get("weight_layout") != layout:
                raise ValueError("Unsupported adapter_rebase format")
            # Preserve selector sizes and RNG draws; appended VQ channels
            # have constant-one gates rather than extra stochastic selectors.
            for network, requested_rank in (("unet", lora_rank_unet_deg), ("vae", lora_rank_vae_deg)):
                selector_rank = rebase[f"selector_rank_{network}"]
                stored_rank = sd[f"rank_{network}_deg"] if sd is not None else requested_rank
                requested_selector = selector_rank_unet if network == "unet" else selector_rank_vae
                if requested_selector is not None and requested_selector != selector_rank:
                    raise ValueError(f"{network}: selector dimension differs from checkpoint")
                if requested_rank not in (selector_rank, stored_rank):
                    raise ValueError(f"Unexpected {network} deg rank for rebased checkpoint: {requested_rank}")
                if stored_rank != selector_rank + rebase[f"fixed_rank_{network}"]:
                    raise ValueError(f"Invalid {network} rank partition in rebased checkpoint")
            if sd is not None:
                lora_rank_unet_deg, lora_rank_vae_deg = sd["rank_unet_deg"], sd["rank_vae_deg"]
        selector_rank_unet = self.adapter_rebase["selector_rank_unet"] if self.adapter_rebase else lora_rank_unet_deg
        selector_rank_vae = self.adapter_rebase["selector_rank_vae"] if self.adapter_rebase else lora_rank_vae_deg

        def adapter_alpha(network, adapter):
            return self.adapter_rebase[f"lora_alpha_{network}_{adapter}"] if self.adapter_rebase else 8

        self.load_vq = load_vq
        if sd is not None and not load_vq:
            for field in ("state_dict_unet", "state_dict_vae"):
                sd[field] = {k: v for k, v in sd[field].items() if ".unet_vq." not in k and ".vae_vq." not in k}
        if sd is not None:
            requested = {"rank_unet_deg": lora_rank_unet_deg, "rank_vae_deg": lora_rank_vae_deg,
                         "rank_unet_vq": lora_rank_unet_vq, "rank_vae_vq": lora_rank_vae_vq}
            for name, rank in requested.items():
                if name.endswith("_vq") and not load_vq:
                    continue
                if name not in sd:
                    raise ValueError(f"Checkpoint is missing {name}; deg+vq inference requires a full restoration checkpoint")
                if rank != sd[name]:
                    raise ValueError(f"{name}: requested {rank}, checkpoint has {sd[name]}")
        self.tokenizer = AutoTokenizer.from_pretrained(sd_path, subfolder="tokenizer", local_files_only=True)
        self.text_encoder = CLIPTextModel.from_pretrained(sd_path, subfolder="text_encoder", local_files_only=True).to(device)
        self.sched = make_1step_sched(sd_path, device)
        self.tau = 1.0

        vae = AutoencoderKL.from_pretrained(sd_path, subfolder="vae", local_files_only=True)
        unet = UNet2DConditionModel.from_pretrained(sd_path, subfolder="unet", local_files_only=True)

        vae.encoder.forward = my_vae_encoder_fwd.__get__(vae.encoder, vae.encoder.__class__)
        vae.decoder.forward = my_vae_decoder_fwd.__get__(vae.decoder, vae.decoder.__class__)
        vae.decoder.skip_conv_1 = torch.nn.Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False).to(device)
        vae.decoder.skip_conv_2 = torch.nn.Conv2d(256, 512, kernel_size=(1, 1), stride=(1, 1), bias=False).to(device)
        vae.decoder.skip_conv_3 = torch.nn.Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False).to(device)
        vae.decoder.skip_conv_4 = torch.nn.Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1), bias=False).to(device)
        vae.decoder.ignore_skip = False

        target_modules_vae = ["conv1", "conv2", "conv_in", "conv_shortcut", "conv", "conv_out",
                "to_k", "to_q", "to_v", "to_out.0",
            ]
        target_modules_unet = [
            "to_k", "to_q", "to_v", "to_out.0", "conv", "conv1", "conv2", "conv_shortcut", "conv_out",
            "proj_in", "proj_out", "ff.net.2", "ff.net.0.proj"
        ]

        self.vae_selector = nn.Sequential(
            nn.Linear(1024, selector_rank_vae * 4),
            nn.GELU(),
            nn.Linear(selector_rank_vae * 4, selector_rank_vae * 2),
            nn.GELU(),
            nn.Linear(selector_rank_vae * 2, selector_rank_vae),
        )

        self.unet_selector = nn.Sequential(
            nn.Linear(1024, selector_rank_unet * 4),
            nn.GELU(),
            nn.Linear(selector_rank_unet * 4, selector_rank_unet * 2),
            nn.GELU(),
            nn.Linear(selector_rank_unet * 2, selector_rank_unet),
        )
        self._linear_initialize_weights()

        if pretrained_path is not None:

            vae_deg_lora_config = LoraConfig(r=sd["rank_vae_deg"], lora_alpha=adapter_alpha("vae", "deg"), init_lora_weights="gaussian", target_modules=sd["vae_lora_target_modules"])
            vae.add_adapter(vae_deg_lora_config, adapter_name="vae_deg")
            if load_vq:
                vae_vq_lora_config = LoraConfig(r=sd["rank_vae_vq"], lora_alpha=adapter_alpha("vae", "vq"), init_lora_weights="gaussian", target_modules=sd["vae_lora_target_modules"])
                vae.add_adapter(vae_vq_lora_config, adapter_name="vae_vq")
            vae.set_adapter(['vae_deg', 'vae_vq'] if load_vq else ['vae_deg'])
            load_restoration_state(vae, sd["state_dict_vae"], lambda k: "lora" in k or "skip_conv" in k)

            unet_deg_lora_config = LoraConfig(r=sd["rank_unet_deg"], lora_alpha=adapter_alpha("unet", "deg"), init_lora_weights="gaussian", target_modules=sd["unet_lora_target_modules"])
            unet.add_adapter(unet_deg_lora_config, adapter_name="unet_deg")
            if load_vq:
                unet_vq_lora_config = LoraConfig(r=sd["rank_unet_vq"], lora_alpha=adapter_alpha("unet", "vq"), init_lora_weights="gaussian", target_modules=sd["unet_lora_target_modules"])
                unet.add_adapter(unet_vq_lora_config, adapter_name="unet_vq")
            unet.set_adapter(['unet_deg', 'unet_vq'] if load_vq else ['unet_deg'])
            load_restoration_state(unet, sd["state_dict_unet"], lambda k: "lora" in k or k.startswith("conv_in."))

            self.vae_selector.load_state_dict(sd["state_dict_vae_selector"], strict=True)
            self.unet_selector.load_state_dict(sd["state_dict_unet_selector"], strict=True)
        else:
            print("Loaded SD-Turbo; initializing fresh deg/vq LoRA, selectors and skip convolutions", flush=True)
            torch.nn.init.constant_(vae.decoder.skip_conv_1.weight, 1e-5)
            torch.nn.init.constant_(vae.decoder.skip_conv_2.weight, 1e-5)
            torch.nn.init.constant_(vae.decoder.skip_conv_3.weight, 1e-5)
            torch.nn.init.constant_(vae.decoder.skip_conv_4.weight, 1e-5)

            vae_deg_lora_config = LoraConfig(r=lora_rank_vae_deg, lora_alpha=adapter_alpha("vae", "deg"), init_lora_weights="gaussian",
                target_modules=target_modules_vae)
            vae.add_adapter(vae_deg_lora_config, adapter_name="vae_deg")
            if load_vq:
                vae_vq_lora_config = LoraConfig(r=lora_rank_vae_vq, lora_alpha=adapter_alpha("vae", "vq"), init_lora_weights="gaussian",
                    target_modules=target_modules_vae)
                vae.add_adapter(vae_vq_lora_config, adapter_name="vae_vq")
            vae.set_adapter(['vae_deg', 'vae_vq'] if load_vq else ['vae_deg'])

            unet_deg_lora_config = LoraConfig(r=lora_rank_unet_deg, lora_alpha=adapter_alpha("unet", "deg"), init_lora_weights="gaussian",
                target_modules=target_modules_unet
            )
            unet.add_adapter(unet_deg_lora_config, adapter_name="unet_deg")
            if load_vq:
                unet_vq_lora_config = LoraConfig(r=lora_rank_unet_vq, lora_alpha=adapter_alpha("unet", "vq"), init_lora_weights="gaussian",
                    target_modules=target_modules_unet)
                unet.add_adapter(unet_vq_lora_config, adapter_name="unet_vq")
            unet.set_adapter(['unet_deg', 'unet_vq'] if load_vq else ['unet_deg'])

        self.lora_rank_unet_deg = lora_rank_unet_deg
        self.lora_rank_vae_deg = lora_rank_vae_deg
        self.lora_rank_unet_vq = lora_rank_unet_vq
        self.lora_rank_vae_vq = lora_rank_vae_vq

        self.target_modules_vae = sd["vae_lora_target_modules"] if sd is not None else target_modules_vae
        self.target_modules_unet = sd["unet_lora_target_modules"] if sd is not None else target_modules_unet

        self.vae_lora_layers = []
        for name, module in vae.named_modules():
            if 'base_layer' in name:
                self.vae_lora_layers.append(name[:-len(".base_layer")])
 
        for name, module in vae.named_modules():
            if name in self.vae_lora_layers:
                module.forward = my_lora_fwd_simple.__get__(module, module.__class__)
                if self.adapter_rebase is not None:
                    module.deg_rebase = {"rank": selector_rank_vae, "vq_scale": adapter_alpha("vae", "vq") / self.adapter_rebase["fixed_rank_vae"]}

        self.unet_lora_layers = []
        for name, module in unet.named_modules():
            if 'base_layer' in name:
                self.unet_lora_layers.append(name[:-len(".base_layer")])

        for name, module in unet.named_modules():
            if name in self.unet_lora_layers:
                module.forward = my_lora_fwd_simple.__get__(module, module.__class__)
                if self.adapter_rebase is not None:
                    module.deg_rebase = {"rank": selector_rank_unet, "vq_scale": adapter_alpha("unet", "vq") / self.adapter_rebase["fixed_rank_unet"]}

        unet.to(device)
        vae.to(device)
        self.unet, self.vae = unet, vae
        self.vae.decoder.gamma = 1
        self.register_buffer("timesteps", torch.tensor([300], device=device, dtype=torch.long), persistent=False)
        self.vae_selector.to(device)
        self.unet_selector.to(device)
        self.rank_vae = selector_rank_vae
        self.rank_unet = selector_rank_unet
        self.text_encoder.requires_grad_(False)

    def _linear_initialize_weights(self):
        for modules in [self.vae_selector, self.unet_selector]:
            for module in modules:
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

    def set_eval(self):
        self.unet.eval()
        self.vae.eval()
        self.vae_selector.eval()
        self.unet_selector.eval()
        self.unet.requires_grad_(False)
        self.vae.requires_grad_(False)

    def set_train(self):
        # Keep frozen backbone weights outside the optimizer and autograd graph.
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
            self.vae.set_adapter(['vae_deg'])
            self.unet.set_adapter(['unet_deg'])
            self.change_weights_requires_grad(self.vae, requires_grad=True, pattern_match_name='vae_deg')
            self.change_weights_requires_grad(self.unet, requires_grad=True, pattern_match_name='unet_deg')
        else:
            self.vae.set_adapter(['vae_deg', 'vae_vq'])
            self.unet.set_adapter(['unet_deg', 'unet_vq'])
            self.change_weights_requires_grad(self.vae, requires_grad=False, pattern_match_name='vae_deg')
            self.change_weights_requires_grad(self.unet, requires_grad=False, pattern_match_name='unet_deg')

    def change_weights_requires_grad(self, model, requires_grad, pattern_match_name):
        for name, parameter in model.named_parameters():
            if pattern_match_name in name:
                parameter.requires_grad_(requires_grad)

    def forward(self, c_t, img_c, deg_c, prompt, classify=False):
        if prompt is not None:
            caption_tokens = self.tokenizer(prompt, max_length=self.tokenizer.model_max_length,
                                            padding="max_length", truncation=True, return_tensors="pt").input_ids.to(c_t.device)
            caption_enc = self.text_encoder(caption_tokens)[0]
        else:
            caption_enc = self.text_encoder(prompt_tokens)[0]

        vae_logits = self.vae_selector(torch.cat([deg_c, img_c], -1))
        unet_logits = self.unet_selector(torch.cat([deg_c, img_c], -1))

        vae_select = Round_Sigmoid_Gumbel(vae_logits, self.tau, True)
        unet_select = Round_Sigmoid_Gumbel(unet_logits, self.tau, True)

        if self.adapter_rebase is not None:
            vae_select = torch.cat((vae_select, vae_select.new_ones(vae_select.shape[0], self.adapter_rebase["fixed_rank_vae"])), dim=-1)
            unet_select = torch.cat((unet_select, unet_select.new_ones(unet_select.shape[0], self.adapter_rebase["fixed_rank_unet"])), dim=-1)

        vae_select_mat = torch.diag_embed(vae_select)
        unet_select_mat = torch.diag_embed(unet_select)

        for layer_name, module in self.vae.named_modules():
            if layer_name in self.vae_lora_layers:
                module.de_mod = vae_select_mat

        for layer_name, module in self.unet.named_modules():
            if layer_name in self.unet_lora_layers:
                module.de_mod = unet_select_mat

        encoded_control = self.vae.encode(c_t).latent_dist.sample() * self.vae.config.scaling_factor
        model_pred = self.unet(encoded_control, self.timesteps, encoder_hidden_states=caption_enc,).sample
        x_denoised = self.sched.step(model_pred, self.timesteps, encoded_control, return_dict=True).prev_sample
        self.vae.decoder.incoming_skip_acts = self.vae.encoder.current_down_blocks
        output_image = (self.vae.decode(x_denoised / self.vae.config.scaling_factor).sample).clamp(-1, 1)

        if classify:
            return output_image, torch.sigmoid(vae_logits), torch.sigmoid(unet_logits)
        else:
            return output_image

    def checkpoint_state(self):
        sd = {}
        if self.adapter_rebase is not None:
            sd["adapter_rebase"] = copy.deepcopy(self.adapter_rebase)
        sd["unet_lora_target_modules"] = self.target_modules_unet
        sd["vae_lora_target_modules"] = self.target_modules_vae
        sd["rank_unet_deg"] = self.lora_rank_unet_deg
        sd["rank_vae_deg"] = self.lora_rank_vae_deg
        sd["rank_unet_vq"] = self.lora_rank_unet_vq
        sd["rank_vae_vq"] = self.lora_rank_vae_vq

        sd["state_dict_unet"] = {k: v for k, v in self.unet.state_dict().items() if "lora" in k or "conv_in" in k}
        sd["state_dict_vae"] = {k: v for k, v in self.vae.state_dict().items() if "lora" in k or "skip_conv" in k}

        sd["state_dict_vae_selector"] = {k: v for k, v in self.vae_selector.state_dict().items()}
        sd["state_dict_unet_selector"] = {k: v for k, v in self.unet_selector.state_dict().items()}
        for field in ("state_dict_unet", "state_dict_vae", "state_dict_vae_selector", "state_dict_unet_selector"):
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
            sd[field] = {k: v for k, v in sd[field].items() if ".vae_vq." not in k and ".unet_vq." not in k}
        torch.save(sd, outf)
