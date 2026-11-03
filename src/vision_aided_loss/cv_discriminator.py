"""CLIP multilevel discriminator from vision-aided-loss, with explicit local weights."""
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import spectral_norm
from .blurpool import BlurPool
from .DiffAugment_pytorch import DiffAugment
from src.open_clip.model import build_model_from_openai_state_dict


class MultiLevelDViT(nn.Module):
    def __init__(self, level=3, in_ch1=768, in_ch2=512, out_ch=256, num_classes=0, activation=nn.LeakyReLU(0.2, inplace=True), down=1):
        super().__init__()

        self.decoder = nn.ModuleList()
        self.level = level
        for _ in range(level-1):
            self.decoder.append(nn.Sequential(
                                BlurPool(in_ch1, pad_type='zero', stride=1, pad_off=1) if down > 1 else nn.Identity(),
                                spectral_norm(nn.Conv2d(in_ch1, out_ch, kernel_size=3, stride=2 if down > 1 else 1, padding=1 if down == 1 else 0)),
                                activation,
                                BlurPool(out_ch, pad_type='zero', stride=1),
                                spectral_norm(nn.Conv2d(out_ch, 1, kernel_size=1, stride=2)))
                                )
        self.decoder.append(nn.Sequential(spectral_norm(nn.Linear(in_ch2, out_ch)), activation))
        self.out = spectral_norm(nn.Linear(out_ch, 1))
        self.embed = None
        if num_classes > 0:
            self.embed = nn.Embedding(num_classes, out_ch)                           

    def forward(self, x, c=None):

        final_pred = []
        for i in range(self.level-1):
            final_pred.append(self.decoder[i](x[i]).squeeze(1))

        h = self.decoder[-1](x[-1].float())
        out = self.out(h)

        if self.embed is not None:
            out += torch.sum(self.embed(c) * h, 1, keepdim=True)

        final_pred.append(out)
        # final_pred = torch.cat(final_pred, 1)
        return final_pred


class CLIPDiscriminator(nn.Module):
    def __init__(self, checkpoint):
        super().__init__()
        clip = torch.jit.load(str(checkpoint), map_location="cpu")
        clip = build_model_from_openai_state_dict(clip.state_dict()).float()
        self.visual = clip.visual.eval().requires_grad_(False)
        self.decoder = MultiLevelDViT()
        self.register_buffer("mean", torch.tensor([0.48145466, 0.4578275, 0.40821073])[None, :, None, None])
        self.register_buffer("std", torch.tensor([0.26862954, 0.26130258, 0.27577711])[None, :, None, None])

    def forward(self, image, for_real=True, for_G=False):
        image = DiffAugment(image, policy="color,translation,cutout")
        image = F.interpolate(image * 0.5 + 0.5, (224, 224), mode="area")
        image = (image - self.mean) / self.std
        visual = self.visual
        x = visual.conv1(image).flatten(2).permute(0, 2, 1)
        token = visual.class_embedding.to(x.dtype)[None, None].expand(x.shape[0], 1, -1)
        x = visual.ln_pre(torch.cat([token, x], 1) + visual.positional_embedding.to(x.dtype))
        x = x.permute(1, 0, 2)
        features = []
        for i, block in enumerate(visual.transformer.resblocks, 1):
            x = block(x)
            if i in (4, 8):
                features.append(x.permute(1, 2, 0)[:, :, 1:].reshape(-1, 768, 7, 7).float())
        features.append((visual.ln_post(x[0]) @ visual.proj).float())
        scores = self.decoder(features)
        target = 0.8 if for_real or for_G else 0.0
        return sum(F.binary_cross_entropy_with_logits(score, torch.full_like(score, target)) for score in scores)
