"""Six-scale SSD head with a measured LiDAR depth reference for nuScenes 3D.

The depth residual and localization-quality prediction are our 3D adaptation;
these are not claimed to be components of the Seeing Through Fog detector.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from .backbone import conv_block
from .boxes import ATTRIBUTES


@torch.no_grad()
def anchor_depth_reference(depth, mask, anchors):
    """Mean log-depth inside the central half of each anchor, with 20m fallback.

    Uses input measurements only. Missing/dropped LiDAR cannot contribute a prior.
    Integral images preserve nearest-surface rasterization and avoid filling gaps
    with artificial depth measurements. The learned residual handles mixed surfaces.
    """
    h, w = depth.shape[-2:]
    center = (anchors[:, :2] + anchors[:, 2:]) * .5
    radius = (anchors[:, 2:] - anchors[:, :2]) * .25
    lo, hi = (center-radius).floor().long(), (center+radius).ceil().long()
    x0, y0 = lo[:, 0].clamp(0, w), lo[:, 1].clamp(0, h)
    x1, y1 = hi[:, 0].clamp(0, w), hi[:, 1].clamp(0, h)
    valid = mask.bool() & torch.isfinite(depth) & (depth > 0)
    def integrate(value):
        integral = F.pad(value[:, 0].cumsum(1).cumsum(2), (1, 0, 1, 0))
        return integral[:, y1, x1]-integral[:, y0, x1]-integral[:, y1, x0]+integral[:, y0, x0]
    count = integrate(valid.float()).clamp_min(0)
    total = integrate(torch.where(valid, depth.float().clamp_min(1).log(), 0.))
    return torch.where(count >= 1, total/count.clamp_min(1), math.log(20.))


class RevisedDetectionHead(nn.Module):
    def __init__(self, in_channels=128, num_classes=10):
        super().__init__()
        self.lateral = nn.ModuleList([nn.Conv2d(in_channels, in_channels, 1) for _ in range(3)])
        self.smooth = nn.ModuleList([conv_block(in_channels, in_channels) for _ in range(3)])
        self.extra = nn.ModuleList([conv_block(in_channels, in_channels, 2) for _ in range(3)])
        self.towers = nn.ModuleList([conv_block(in_channels, in_channels) for _ in range(6)])
        self.predictors = nn.ModuleDict({k: nn.Conv2d(in_channels, 6*n, 3, padding=1)
            for k,n in [('logits', num_classes+1), ('boxes', 4), ('boxes3d', 10),
                        ('attributes', len(ATTRIBUTES)), ('quality', 1)]})
        for layer in self.predictors.values():
            nn.init.normal_(layer.weight, std=.01)
            nn.init.zeros_(layer.bias)
        with torch.no_grad():
            bias = self.predictors['boxes3d'].bias.view(6, 10)
            bias[:, 3:6] = torch.tensor([1.8, 4., 1.6]).log()
            bias[:, 7] = 1
            self.predictors['logits'].bias.view(6, num_classes+1)[:, 0] = 2

    def forward(self, features, image_hw, depth, depthmask):
        lateral = [layer(x) for layer,x in zip(self.lateral, features)]
        for i in (1, 0):
            lateral[i] = lateral[i] + F.interpolate(lateral[i+1], size=lateral[i].shape[-2:], mode='nearest')
        pyramid = [layer(x) for layer,x in zip(self.smooth, lateral)]
        for layer in self.extra:
            pyramid.append(layer(pyramid[-1]))
        result = {k: [] for k in self.predictors}
        anchors = []
        ih, iw = image_hw
        for level, (feature,tower) in enumerate(zip(pyramid,self.towers)):
            x = tower(feature)
            b,_,h,w = x.shape
            for key,layer in self.predictors.items():
                value = layer(x)
                result[key].append(value.view(b,6,-1,h,w).permute(0,3,4,1,2).reshape(b,h*w*6,-1))
            y,xx = torch.meshgrid(torch.arange(h,device=x.device),torch.arange(w,device=x.device),indexing='ij')
            center = torch.stack(((xx+.5)*iw/w,(y+.5)*ih/h),-1).float().reshape(-1,1,2)
            base = min(ih,iw)*(.04,.09,.18,.32,.55,.8)[level]
            sizes = torch.tensor([[base*s*math.sqrt(r),base*s/math.sqrt(r)]
                for s in (1.,math.sqrt(2)) for r in (.5,1.,2.)],device=x.device,dtype=torch.float32)
            anchors.append(torch.cat((center-sizes/2,center+sizes/2),-1).reshape(-1,4))
        output = {k: torch.cat(v,1) for k,v in result.items()}
        output['anchors'] = torch.cat(anchors)
        prior = anchor_depth_reference(depth,depthmask,output['anchors'])
        raw = output['boxes3d']
        output['boxes3d'] = torch.cat((raw[:,:,:2],raw[:,:,2:3]+prior[:,:,None],raw[:,:,3:]),-1)
        output['revised'] = True
        return output
