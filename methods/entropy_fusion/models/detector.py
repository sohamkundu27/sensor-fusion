"""Three-stream, entropy-gated image-plane detector with metric 3D regression."""
import torch
from torch import nn
from .backbone import CameraBackbone, DepthBackbone
from .entropy import PatchEntropy
from .fusion import FeatureExchange, drop_modalities
from .head import DetectionHead


class EntropyFusionDetector(nn.Module):
    def __init__(self, pretrained=True, modality_dropout=.5):
        super().__init__()
        self.modality_dropout = modality_dropout
        self.camera = CameraBackbone(pretrained)
        self.lidar, self.radar = DepthBackbone(), DepthBackbone()
        self.entropy = PatchEntropy()
        self.exchange = nn.ModuleList([FeatureExchange(channels) for channels in
                                       zip(self.camera.channels, self.lidar.channels, self.radar.channels)])
        self.head = DetectionHead()
        self.register_buffer('mean', torch.tensor([.485, .456, .406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([.229, .224, .225]).view(1, 3, 1, 1))

    def forward(self, batch):
        inputs = [batch[k] for k in ('camera', 'lidar', 'radar')]
        masks = [batch[k+'_mask'] for k in ('camera', 'lidar', 'radar')]
        inputs, masks, available = drop_modalities(inputs, masks, self.modality_dropout, self.training)
        entropies = [self.entropy(x, m) for x, m in zip(inputs, masks)]
        inputs[0] = ((inputs[0]-self.mean)/self.std) * masks[0]
        streams = [net(x) for net, x in zip((self.camera, self.lidar, self.radar), inputs)]
        fused = [exchange(features, entropies, available) for exchange, features in zip(self.exchange, zip(*streams))]
        result = self.head(fused, inputs[0].shape[-2:])
        result['available'] = available
        return result
