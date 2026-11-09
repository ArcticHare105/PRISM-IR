import torch
from torch import nn


class proto_DegraCls(nn.Module):
    def __init__(self, lora_rank_vae, lora_rank_unet, class_num=10):
        super().__init__()
        self.vae_prototypes = nn.Parameter(
            self._orthogonal_prototypes(class_num, lora_rank_vae)
        )
        self.unet_prototypes = nn.Parameter(
            self._orthogonal_prototypes(class_num, lora_rank_unet)
        )

    def _orthogonal_prototypes(self, class_num, feature_dim):
        if class_num > feature_dim:
            raise ValueError("Prototype count cannot exceed feature dimensions")
        return nn.init.orthogonal_(torch.empty(class_num, feature_dim))

    def save_model(self, outf):
        torch.save(
            {
                "state_dict_vae_prototypes": self.vae_prototypes.detach().cpu(),
                "state_dict_unet_prototypes": self.unet_prototypes.detach().cpu(),
            },
            outf,
        )

    def forward(self, vae_select, unet_select):
        vae_proto = torch.sigmoid(self.vae_prototypes)
        unet_proto = torch.sigmoid(self.unet_prototypes)
        vae_cls = torch.sqrt(
            torch.sum((vae_select.unsqueeze(1) - vae_proto.unsqueeze(0)) ** 2, dim=2)
            + 1e-6
        )
        unet_cls = torch.sqrt(
            torch.sum((unet_select.unsqueeze(1) - unet_proto.unsqueeze(0)) ** 2, dim=2)
            + 1e-6
        )
        return vae_cls, unet_cls, vae_proto, unet_proto
