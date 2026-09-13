"""Entropy-conditioned gates with concatenation at each backbone resolution."""
import torch
from torch import nn
from torch.nn import functional as F
from .backbone import conv_block


class FeatureExchange(nn.Module):
    def __init__(self, channels, out_channels=96):
        super().__init__()
        self.gates = nn.ModuleList([nn.Conv2d(2, c, 3, padding=1) for c in channels])
        self.project = conv_block(sum(channels), out_channels)

    def forward(self, features, entropies, available):
        gated = []
        for i, (feature, entropy, gate) in enumerate(zip(features, entropies, self.gates)):
            condition = F.interpolate(entropy, size=feature.shape[-2:], mode='bilinear', align_corners=False)
            # Prevent convolution/normalization biases from reviving a dropped stream.
            weight = gate(condition).sigmoid() * available[:, i, None, None, None]
            gated.append(feature * weight)
        return self.project(torch.cat(gated, dim=1))


def drop_modalities(inputs, masks, probability, training):
    """With probability p per sample, drop exactly one uniformly chosen stream."""
    if not 0 <= probability <= 1:
        raise ValueError('Dropout probability must be between zero and one')
    batch = inputs[0].shape[0]
    available = torch.stack([m.flatten(1).any(1) for m in masks], dim=1)
    if training and probability:
        # Never intentionally remove the final naturally available modality.
        eligible = (available.sum(1) > 1) & (torch.rand(batch, device=inputs[0].device) < probability)
        choice = torch.randint(3, (batch,), device=inputs[0].device)
        keep = torch.ones_like(available)
        keep[torch.arange(batch, device=choice.device), choice] = ~eligible
        available = available & keep
    xs = [x * available[:, i, None, None, None] for i, x in enumerate(inputs)]
    ms = [m & available[:, i, None, None, None] for i, m in enumerate(masks)]
    return xs, ms, available
