"""Progressive entropy-steered image-plane fusion, adapted to nuScenes 3D.

The paper's exchange acts inside feature extraction. Each resolution here sends
the joint sensor context back into every surviving stream before its next stage.
ResNet18 pretraining, coverage channels, residual exchange and the 3D head are
explicit nuScenes adaptations rather than a claim of an exact VGG reproduction.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from .backbone import CameraBackbone, conv_block
from .fusion import drop_modalities


class MeasurementEntropy(nn.Module):
    """256-bin entropy per input channel, plus measured coverage, in 16x16 tiles.

Missing measurements are encoded as zero and included in the histogram, as in
the paper's image representation. Coverage additionally distinguishes missing
data from a measured constant signal. DHI channels are never treated as RGB.
"""
    channels = 4

    def __init__(self, patch_size=16):
        super().__init__()
        if patch_size < 1:
            raise ValueError('Entropy patch size must be positive')
        self.patch_size = patch_size

    @torch.no_grad()
    def forward(self, value, mask):
        if value.ndim != 4 or value.shape[1] != 3:
            raise ValueError('Expected three image or sensor channels')
        p = self.patch_size
        if value.shape[-2] % p or value.shape[-1] % p:
            raise ValueError('Input dimensions must be divisible by entropy patch size')
        mask = mask.bool()
        value = value.float().masked_fill(~mask, 0)
        tiles = value.unfold(2, p, p).unfold(3, p, p).flatten(-2)
        indices = (tiles.clamp(0, 1) * 255).round().long()
        counts = torch.zeros(*indices.shape[:-1], 256, device=value.device)
        counts.scatter_add_(-1, indices, torch.ones_like(indices, dtype=torch.float32))
        probabilities = counts / (p * p)
        entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(-1) / math.log(256)
        coverage = F.avg_pool2d(mask.float(), p, p)
        return torch.cat((entropy, coverage), dim=1)


def drop_independent_modalities(inputs, masks, probability, training):
    """Independent sensor dropout conditioned on retaining a measured stream.

The paper states p=0.5 sensor dropout but does not specify how simultaneous
missing sensors are handled. We sample the exact conditional distribution over
the seven nonempty subsets. At p=1 its limit keeps one available sensor uniformly.
An already empty item remains empty rather than inventing measurements.
"""
    if len(inputs) != 3 or len(masks) != 3 or not 0 <= probability <= 1:
        raise ValueError('Expected three modalities and a probability in [0, 1]')
    available = torch.stack([m.flatten(1).any(1) for m in masks], dim=1)
    if training and probability:
        subsets = torch.tensor([[bool(bits & (1 << i)) for i in range(3)]
                                for bits in range(1, 8)], device=available.device)
        permitted = ~(subsets[None] & ~available[:, None]).any(-1)
        kept = subsets.sum(-1)[None]
        dropped = (available.sum(-1)[:, None] - kept).clamp_min(0)
        if probability == 1:
            weights = (permitted & (kept == 1)).float()
        else:
            weights = permitted.float() * (1 - probability) ** kept * probability ** dropped
        empty = ~available.any(-1)
        weights[empty, 0] = 1
        selected = subsets[torch.multinomial(weights, 1).squeeze(1)]
        available = available & selected
    xs = [x.masked_fill(~available[:, i, None, None, None], 0) for i, x in enumerate(inputs)]
    ms = [m.bool() & available[:, i, None, None, None] for i, m in enumerate(masks)]
    return xs, ms, available


class SensorBackbone(nn.Module):
    channels = (32, 64, 128)

    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(conv_block(3, 16, 2), conv_block(16, 16, 2))
        self.stages = nn.ModuleList([nn.Sequential(conv_block(a, b, 2), conv_block(b, b))
                                    for a, b in zip((16, 32, 64), self.channels)])


class ProgressiveFeatureExchange(nn.Module):
    """Joint entropy gates, entropy concatenation, and residual stream feedback."""
    def __init__(self, channels, out_channels=128, fusion_mode='entropy', residual_scale=.1):
        super().__init__()
        if fusion_mode not in ('entropy', 'concat'):
            raise ValueError('fusion_mode must be entropy or concat')
        self.fusion_mode = fusion_mode
        self.residual_scale = residual_scale
        self.channels = tuple(channels)
        entropy_channels = 3 * MeasurementEntropy.channels
        self.gate = nn.Conv2d(entropy_channels, sum(channels), 3, padding=1) if fusion_mode == 'entropy' else None
        self.project = conv_block(sum(channels) + (entropy_channels if self.gate is not None else 0), out_channels)
        self.feedback = nn.ModuleList([nn.Conv2d(out_channels, c, 1, bias=False) for c in channels])

    def forward(self, features, entropies, available):
        # Frozen-BN camera activations can exceed FP16 range before normalization.
        # Keep both the joint convolution and residual exchange in FP32.
        with torch.autocast(device_type=features[0].device.type, enabled=False):
            features = [x.float().masked_fill(~available[:, i, None, None, None], 0) for i, x in enumerate(features)]
            joint = torch.cat(features, dim=1)
            if self.gate is not None:
                condition = torch.cat([F.interpolate(entropy, size=joint.shape[-2:], mode='bilinear', align_corners=False)
                                       * available[:, i, None, None, None] for i, entropy in enumerate(entropies)], dim=1)
                joint = torch.cat((joint * self.gate(condition).sigmoid(), condition), dim=1)
            fused = self.project(joint)
            fused = fused.masked_fill(~available.any(1)[:, None, None, None], 0)
            updated = [(feature + self.residual_scale * feedback(fused)).masked_fill(
                        ~available[:, i, None, None, None], 0)
                       for i, (feature, feedback) in enumerate(zip(features, self.feedback))]
            return updated, fused


class ReimplementedEntropyFusionDetector(nn.Module):
    def __init__(self, pretrained=True, modality_dropout=.5, fusion_mode='entropy', dropout_mode='independent', num_classes=10):
        super().__init__()
        from .revised_head import RevisedDetectionHead
        self.modality_dropout = modality_dropout
        self.dropout_mode = dropout_mode
        if dropout_mode not in ('single','independent'):
            raise ValueError('Unknown modality dropout mode')
        self.fusion_mode = fusion_mode
        self.camera = CameraBackbone(pretrained)
        self.lidar, self.radar = SensorBackbone(), SensorBackbone()
        self.entropy = MeasurementEntropy()
        self.exchange = nn.ModuleList([ProgressiveFeatureExchange(channels, fusion_mode=fusion_mode)
                                       for channels in zip(self.camera.channels, self.lidar.channels, self.radar.channels)])
        self.head = RevisedDetectionHead(in_channels=128, num_classes=num_classes)
        self.register_buffer('mean', torch.tensor([.485, .456, .406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([.229, .224, .225]).view(1, 3, 1, 1))

    def forward(self, batch):
        inputs = [batch[k] for k in ('camera', 'lidar', 'radar')]
        masks = [batch[k + '_mask'] for k in ('camera', 'lidar', 'radar')]
        dropout = drop_modalities if self.dropout_mode == 'single' else drop_independent_modalities
        inputs, masks, available = dropout(inputs, masks, self.modality_dropout, self.training)
        entropies = [self.entropy(x, m) for x, m in zip(inputs, masks)] if self.fusion_mode == 'entropy' else None
        inputs[0] = ((inputs[0] - self.mean) / self.std) * masks[0]
        networks = (self.camera, self.lidar, self.radar)
        def run_stage(module, value, camera=False):
            if camera:
                # Prevent overflowing camera features before they reach exchange.
                with torch.autocast(device_type=value.device.type, enabled=False):
                    return module(value.float())
            return module(value)
        streams = [run_stage(net.stem, x, camera=i == 0).masked_fill(~available[:, i, None, None, None], 0)
                   for i, (net, x) in enumerate(zip(networks, inputs))]
        fused = []
        for level, exchange in enumerate(self.exchange):
            streams = [run_stage(net.stages[level], x, camera=i == 0) for i, (net, x) in enumerate(zip(networks, streams))]
            streams, context = exchange(streams, entropies, available)
            fused.append(context)
        # Physical depth must follow sensor dropout too; otherwise the head can
        # bypass the dropped LiDAR branch through its geometric reference.
        depth = batch['lidar_depth_m'].masked_fill(~available[:, 1, None, None, None], 0)
        result = self.head(fused, inputs[0].shape[-2:], depth, masks[1])
        center = (result['anchors'][:, :2] + result['anchors'][:, 2:]) / 2
        h, w = inputs[0].shape[-2:]
        result['valid_anchors'] = batch['camera_mask'][:, 0, center[:, 1].long().clamp(0, h - 1),
                                                      center[:, 0].long().clamp(0, w - 1)]
        result['available'] = available
        result['metric_suppression'] = getattr(self, 'metric_suppression', True)
        result['revised'] = True
        return result
