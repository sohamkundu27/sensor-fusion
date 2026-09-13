"""ImageNet ResNet18 and two compact depth-image CNNs, strides 8/16/32."""
from torch import nn
from torchvision.models import resnet18, ResNet18_Weights


def conv_block(cin, cout, stride=1):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, stride, 1, bias=False),
                         nn.GroupNorm(8, cout), nn.ReLU(inplace=True))


class CameraBackbone(nn.Module):
    channels = (128, 256, 512)

    def __init__(self, pretrained=True):
        super().__init__()
        net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool, net.layer1)
        self.stages = nn.ModuleList([net.layer2, net.layer3, net.layer4])
        self.train()

    def train(self, mode=True):
        super().train(mode)
        # Preserve ImageNet running statistics at small batch sizes; affine parameters train.
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
        return self

    def forward(self, x):
        x = self.stem(x)
        outputs = []
        for stage in self.stages:
            x = stage(x)
            outputs.append(x)
        return outputs


class DepthBackbone(nn.Module):
    channels = (32, 64, 128)

    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(conv_block(1, 16, 2), conv_block(16, 16, 2))
        self.stages = nn.ModuleList([nn.Sequential(conv_block(a, b, 2), conv_block(b, b))
                                    for a, b in zip((16, 32, 64), self.channels)])

    def forward(self, x):
        x = self.stem(x)
        outputs = []
        for stage in self.stages:
            x = stage(x)
            outputs.append(x)
        return outputs
