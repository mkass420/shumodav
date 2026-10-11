import torch.nn as nn


class DnCNNLite(nn.Module):
    """Conv+ReLU → (Conv+BN+ReLU)×(depth-2) → Conv; residual learning: x - net(x)."""

    def __init__(self, depth: int = 10, channels: int = 48):
        super().__init__()
        layers = [nn.Conv2d(1, channels, 3, padding=1), nn.ReLU(inplace=True)]
        for _ in range(depth - 2):
            layers += [nn.Conv2d(channels, channels, 3, padding=1, bias=False),
                       nn.BatchNorm2d(channels), nn.ReLU(inplace=True)]
        layers.append(nn.Conv2d(channels, 1, 3, padding=1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return x - self.net(x)
