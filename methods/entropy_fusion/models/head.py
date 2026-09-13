"""SSD-style anchors with class, image box and camera-ray 3D predictions."""
import math
import torch
from torch import nn
from .backbone import conv_block
from .boxes import ATTRIBUTES


class DetectionHead(nn.Module):
    def __init__(self, channels=96, num_classes=10):
        super().__init__()
        self.tower = conv_block(channels, channels)
        self.predictors = nn.ModuleDict({key: nn.Conv2d(channels, 6*n, 3, padding=1)
                                        for key, n in [('logits', num_classes+1), ('boxes', 4),
                                                       ('boxes3d', 10), ('attributes', len(ATTRIBUTES))]})
        for layer in self.predictors.values():
            nn.init.normal_(layer.weight, std=.01)
            nn.init.zeros_(layer.bias)
        # Sensible metric initialization, independently trainable for every anchor.
        with torch.no_grad():
            bias = self.predictors['boxes3d'].bias.view(6, 10)
            bias[:, 2] = math.log(20)
            bias[:, 3:6] = torch.tensor([1.8, 4., 1.6]).log()
            bias[:, 7] = 1
            self.predictors['logits'].bias.view(6, num_classes+1)[:, 0] = 2

    def forward(self, features, image_hw):
        result = {k: [] for k in self.predictors}
        anchors = []
        ih, iw = image_hw
        for level, feature in enumerate(features):
            x = self.tower(feature)
            b, _, h, w = x.shape
            for key, layer in self.predictors.items():
                value = layer(x)
                result[key].append(value.view(b, 6, -1, h, w).permute(0, 3, 4, 1, 2).reshape(b, h*w*6, -1))
            y, xx = torch.meshgrid(torch.arange(h, device=x.device), torch.arange(w, device=x.device), indexing='ij')
            center = torch.stack(((xx+.5)*iw/w, (y+.5)*ih/h), -1).float().reshape(-1, 1, 2)
            base = min(ih, iw) * (.1, .25, .5)[level]
            sizes = x.new_tensor([[base*s*math.sqrt(r), base*s/math.sqrt(r)]
                                  for s in (1., math.sqrt(2)) for r in (.5, 1., 2.)]).float()
            anchors.append(torch.cat((center-sizes/2, center+sizes/2), -1).reshape(-1, 4))
        return {**{k: torch.cat(v, 1) for k, v in result.items()}, 'anchors': torch.cat(anchors)}
