import math
from typing import Any
import torch
from torch.nn import functional as F
from diffusers import DDPMScheduler


def make_1step_sched(pretrained_path, device="cuda"):
    scheduler = DDPMScheduler.from_pretrained(
        pretrained_path, subfolder="scheduler", local_files_only=True
    )
    scheduler.set_timesteps(1, device=device)
    scheduler.alphas_cumprod = scheduler.alphas_cumprod.to(device)
    return scheduler


def soft_target_ce_loss(logits, target, num_classes=None):
    if target.dim() == 1:
        assert num_classes is not None, "num_classes must be provided for hard targets."
        target = F.one_hot(target, num_classes=num_classes).float()

    log_probs = F.log_softmax(logits, dim=1)

    loss = -(target.detach() * log_probs).sum(dim=1).mean()
    return loss


def contrastive_select_loss(select_vec, class_num):
    B, L = select_vec.shape

    k = L // class_num
    topk_values, _ = torch.topk(select_vec, k, dim=1)
    topk_mean = topk_values.mean(dim=1)

    bottomk_values, _ = torch.topk(-select_vec, k * 2, dim=1)
    bottomk_mean = -bottomk_values.mean(dim=1)

    loss = 1.0 + (bottomk_mean - topk_mean).mean()

    return loss


def ssim_loss(img1, img2, window_size=11, size_average=True):
    def _gaussian(window_size, sigma):
        gauss = torch.tensor(
            [
                (1 / (sigma * math.sqrt(2.0 * torch.pi)))
                * math.exp(-((x - window_size // 2) ** 2) / (2 * sigma**2))
                for x in range(window_size)
            ]
        )
        return gauss / gauss.sum()

    def _create_gaussian_window(window_size, channel, sigma=1.5):
        _1d_window = _gaussian(window_size, sigma).unsqueeze(1)
        _2d_window = _1d_window @ _1d_window.T
        _2d_window = _2d_window.float().unsqueeze(0).unsqueeze(0)
        window = _2d_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    device = img1.device
    channel = img1.size(1)

    window = _create_gaussian_window(window_size, channel).to(device)

    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = (
        F.conv2d(img1**2, window, padding=window_size // 2, groups=channel) - mu1_sq
    )
    sigma2_sq = (
        F.conv2d(img2**2, window, padding=window_size // 2, groups=channel) - mu2_sq
    )
    sigma12 = (
        F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel)
        - mu1_mu2
    )

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )
    ssim_loss_value = 1 - ssim_map.mean()

    if size_average:
        return ssim_loss_value
    else:
        return 1 - ssim_map.mean([1, 2, 3])


def my_vae_encoder_fwd(self, sample):
    sample = self.conv_in(sample)
    l_blocks = []

    for down_block in self.down_blocks:
        l_blocks.append(sample)
        sample = down_block(sample)

    sample = self.mid_block(sample)
    sample = self.conv_norm_out(sample)
    sample = self.conv_act(sample)
    sample = self.conv_out(sample)
    self.current_down_blocks = l_blocks
    return sample


def my_vae_decoder_fwd(self, sample, latent_embeds=None):
    sample = self.conv_in(sample)
    upscale_dtype = next(iter(self.up_blocks.parameters())).dtype

    sample = self.mid_block(sample, latent_embeds)
    sample = sample.to(upscale_dtype)
    if not self.ignore_skip:
        skip_convs = [
            self.skip_conv_1,
            self.skip_conv_2,
            self.skip_conv_3,
            self.skip_conv_4,
        ]

        for idx, up_block in enumerate(self.up_blocks):
            skip_in = skip_convs[idx](self.incoming_skip_acts[::-1][idx] * self.gamma)

            sample = sample + skip_in
            sample = up_block(sample, latent_embeds)
    else:
        for idx, up_block in enumerate(self.up_blocks):
            sample = up_block(sample, latent_embeds)

    if latent_embeds is None:
        sample = self.conv_norm_out(sample)
    else:
        sample = self.conv_norm_out(sample, latent_embeds)
    sample = self.conv_act(sample)
    sample = self.conv_out(sample)
    return sample


def my_lora_fwd_simple(
    self, x: torch.Tensor, *args: Any, **kwargs: Any
) -> torch.Tensor:
    self._check_forward_args(x, *args, **kwargs)
    adapter_names = kwargs.pop("adapter_names", None)

    if self.disable_adapters:
        if self.merged:
            self.unmerge()
        result = self.base_layer(x, *args, **kwargs)
    elif adapter_names is not None:
        result = self._mixed_batch_forward(
            x, *args, adapter_names=adapter_names, **kwargs
        )
    elif self.merged:
        result = self.base_layer(x, *args, **kwargs)
    else:
        result = self.base_layer(x, *args, **kwargs)
        torch_result_dtype = result.dtype
        for active_adapter in self.active_adapters:
            if active_adapter not in self.lora_A.keys():
                continue
            lora_A = self.lora_A[active_adapter]
            lora_B = self.lora_B[active_adapter]
            dropout = self.lora_dropout[active_adapter]
            scaling = self.scaling[active_adapter]
            x = x.to(lora_A.weight.dtype)

            if not self.use_dora[active_adapter]:
                _tmp = lora_A(dropout(x))

                if "deg" in active_adapter:
                    if isinstance(lora_A, torch.nn.Conv2d):
                        _tmp = torch.einsum("...khw,...kr->...rhw", _tmp, self.de_mod)
                    elif isinstance(lora_A, torch.nn.Linear):
                        _tmp = torch.einsum("...lk,...kr->...lr", _tmp, self.de_mod)
                    else:
                        raise NotImplementedError(
                            "only conv and linear are supported yet."
                        )

                result = result + lora_B(_tmp) * scaling
            else:
                x = dropout(x)
                result = result + self._apply_dora(
                    x, lora_A, lora_B, scaling, active_adapter
                )

        result = result.to(torch_result_dtype)

    return result
