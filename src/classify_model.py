"""Training-only selector prototypes, extracted from the original IR code."""
import torch
from torch import nn
from torch.nn import functional as F


class proto_DegraCls(nn.Module):
    def __init__(
        self,
        lora_rank_vae,
        lora_rank_unet,
        class_num=10,
        pretrained_path=None
    ):
        super().__init__()
        self.vae_prototypes = nn.Parameter(self._orthogonal_prototypes(class_num, lora_rank_vae))
        self.unet_prototypes = nn.Parameter(self._orthogonal_prototypes(class_num, lora_rank_unet))

        if pretrained_path is not None:
            sd = torch.load(pretrained_path, map_location="cpu")
            self.vae_prototypes = nn.Parameter(sd["state_dict_vae_prototypes"])
            self.unet_prototypes = nn.Parameter(sd["state_dict_unet_prototypes"])

    def _orthogonal_prototypes(self, class_num, feature_dim):
        W = torch.randn(class_num, feature_dim)
        for i in range(1, class_num):
            for j in range(i):
                W[i] -= torch.dot(W[i], W[j]) * W[j]
            W[i] = W[i] / W[i].norm()
        return W

    def set_eval(self):
        self.vae_prototypes.requires_grad_(False)
        self.unet_prototypes.requires_grad_(False)

    def set_train(self):
        self.vae_prototypes.requires_grad_(True)
        self.unet_prototypes.requires_grad_(True)

    def save_model(self, outf):
        sd = {}
        sd['state_dict_vae_prototypes'] = self.vae_prototypes
        sd['state_dict_unet_prototypes'] = self.unet_prototypes
        torch.save(sd, outf)

    def cal_similarity(self, inp, proto, scale=2.0):
        norm_inp = F.normalize(inp, p=2, dim=1)
        norm_proto = F.normalize(proto, p=2, dim=1)

        logits = torch.mm(norm_inp, norm_proto.t()) * scale
        return logits

    def cal_proto_loss(self, proto, lambda_sparse=0.1, lambda_ortho=0.1):
        l1_loss = lambda_sparse * torch.norm(proto, p=1, dim=1).mean()
        
        norm_proto = proto / (torch.norm(proto, dim=1, keepdim=True) + 1e-6)
        similarity = torch.mm(norm_proto, norm_proto.t())
        mask = ~torch.eye(similarity.shape[0], dtype=bool)
        ortho_loss = lambda_ortho * similarity[mask].mean()
        return l1_loss + ortho_loss

    def forward(self, vae_select, unet_select):
        vae_cls = torch.sqrt(torch.sum((vae_select.unsqueeze(1) - \
            torch.sigmoid(self.vae_prototypes).unsqueeze(0)) ** 2, dim=2) + 1e-6)
        unet_cls = torch.sqrt(torch.sum((unet_select.unsqueeze(1) -\
            torch.sigmoid(self.unet_prototypes).unsqueeze(0)) ** 2, dim=2) + 1e-6)

        # vae_cls = self.cal_similarity(vae_select, self.vae_prototypes)
        # unet_cls = self.cal_similarity(unet_select, self.unet_prototypes)

        # vae_proto_loss = self.cal_proto_loss(self.vae_prototypes)
        # unet_proto_loss = self.cal_proto_loss(self.unet_prototypes)

        return vae_cls, unet_cls, torch.sigmoid(self.vae_prototypes), torch.sigmoid(self.unet_prototypes)
