from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DehazeUNet(nn.Module):
    def __init__(self, base_channels: int = 32) -> None:
        super().__init__()
        c = base_channels
        self.enc1 = ConvBlock(3, c)
        self.enc2 = ConvBlock(c, c * 2)
        self.enc3 = ConvBlock(c * 2, c * 4)
        self.bottleneck = ConvBlock(c * 4, c * 8)

        self.up3 = nn.ConvTranspose2d(c * 8, c * 4, 2, stride=2)
        self.dec3 = ConvBlock(c * 8, c * 4)
        self.up2 = nn.ConvTranspose2d(c * 4, c * 2, 2, stride=2)
        self.dec2 = ConvBlock(c * 4, c * 2)
        self.up1 = nn.ConvTranspose2d(c * 2, c, 2, stride=2)
        self.dec1 = ConvBlock(c * 2, c)

        self.out = nn.Sequential(
            nn.Conv2d(c, 3, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[-2:]
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))
        b = self.bottleneck(F.max_pool2d(e3, 2))

        d3 = self.up3(b)
        d3 = self.dec3(torch.cat([d3, self._match(e3, d3)], dim=1))
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, self._match(e2, d2)], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, self._match(e1, d1)], dim=1))
        out = self.out(d1)
        if out.shape[-2:] != input_size:
            out = F.interpolate(out, size=input_size, mode="bilinear", align_corners=False)
        return out

    @staticmethod
    def _match(skip: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if skip.shape[-2:] == target.shape[-2:]:
            return skip
        return F.interpolate(skip, size=target.shape[-2:], mode="bilinear", align_corners=False)


def build_model(base_channels: int = 32) -> DehazeUNet:
    return DehazeUNet(base_channels=base_channels)
