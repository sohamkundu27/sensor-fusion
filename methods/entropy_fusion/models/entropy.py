"""Masked, normalized Shannon entropy on non-overlapping 16-pixel patches."""
import math
import torch
from torch import nn


class PatchEntropy(nn.Module):
    def __init__(self, patch_size=16, bins=16):
        super().__init__()
        if patch_size < 1 or bins < 2:
            raise ValueError('Positive patch size and at least two bins required')
        self.patch_size, self.bins = patch_size, bins

    @torch.no_grad()
    def forward(self, value, mask):
        # Camera uses luminance; depth maps are already scaled to [0, 1].
        value = value.float()
        if value.shape[1] == 3:
            value = (value * value.new_tensor([.299, .587, .114])[None, :, None, None]).sum(1, keepdim=True)
        p = self.patch_size
        if value.shape[-2] % p or value.shape[-1] % p:
            raise ValueError('Input dimensions must be divisible by patch size')
        tiles = value.unfold(2, p, p).unfold(3, p, p).flatten(-2)
        valid = mask.bool().unfold(2, p, p).unfold(3, p, p).flatten(-2)
        indices = (tiles.clamp(0, 1) * self.bins).long().clamp_max(self.bins - 1)
        counts = torch.zeros(*indices.shape[:-1], self.bins, device=value.device)
        counts.scatter_add_(-1, indices, valid.float())
        total = counts.sum(-1, keepdim=True)
        prob = counts / total.clamp_min(1)
        entropy = -(prob * prob.clamp_min(1e-12).log()).sum(-1) / math.log(self.bins)
        coverage = total.squeeze(-1) / (p * p)
        # Coverage distinguishes an empty sensor from a measured constant patch.
        return torch.cat((entropy, coverage), dim=1)
