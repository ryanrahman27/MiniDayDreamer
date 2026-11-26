"""End-to-end camera conditioning for the three-command flow-matching policy."""

import torch
from torch import nn

from .models import RelativeFlowPolicy


class CameraEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        layers = []
        for cin, cout, k, s in [
            (3, 24, 5, 2),
            (24, 32, 3, 2),
            (32, 48, 3, 2),
            (48, 48, 3, 1),
        ]:
            layers += [
                nn.Conv2d(cin, cout, k, s, k // 2),
                nn.GroupNorm(8, cout),
                nn.SiLU(),
            ]
        self.camera = nn.Sequential(*layers)
        self.heatmaps = nn.Conv2d(48, 24, 1)
        self.context = nn.Sequential(
            nn.Conv2d(48, 16, 3, 2, 1), nn.SiLU(), nn.AdaptiveAvgPool2d((3, 3))
        )
        self.head = nn.Sequential(
            nn.Linear(2 * (48 + 144), 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Linear(128, 16),
            nn.LayerNorm(16),
        )
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, 12), torch.linspace(-1, 1, 12), indexing="ij"
        )
        self.register_buffer("grid", torch.stack((xx, yy), -1).reshape(144, 2))

    def forward(self, rgb):
        if rgb.ndim != 4 or rgb.shape[1:] != (6, 96, 96) or rgb.dtype != torch.uint8:
            raise ValueError("Expected table and wrist RGB bytes [B,6,96,96]")
        images = rgb.reshape(-1, 3, 96, 96).float() / 255
        if self.training:
            gain = torch.empty_like(images[:, :, :1, :1]).uniform_(0.88, 1.12)
            bias = torch.empty_like(images[:, :1, :1, :1]).uniform_(-0.035, 0.035)
            images = (images * gain + bias).clamp(0, 1)
        fmap = self.camera(images)
        weights = (self.heatmaps(fmap) * 4).flatten(2).softmax(-1)
        xy = (weights @ self.grid).flatten(1)
        raw = self.head(
            torch.cat((xy, self.context(fmap).flatten(1)), 1).reshape(len(rgb), -1)
        )
        return raw


class VisualFlowHead(RelativeFlowPolicy):
    """Use the original flow architecture and solver on 16 learned visual features."""

    def condition(self, obs):
        return obs[:, :16]
