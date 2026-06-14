"""Convolutional building blocks shared across the repo.

``ResidualBlock`` is the *exact* pattern used by raccoon's trunk
(``raccoon/model/network.py``): two 3x3 convolutions with BatchNorm, wrapped by
a skip connection. ``ConvTrunk`` stacks an input convolution and several
residual blocks; it is reused by:

* :class:`azl.foundations.models.SmallResNet`     (A2 -- ResNet on MNIST)
* :class:`azl.foundations.models.TwoHeadedNet`    (A4 -- multi-output net)
* :class:`azl.network.AZNet`                       (Track B -- AlphaZero net)

so the very block you train on digits is the block that powers the game net.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Conv3x3 -> BN -> ReLU -> Conv3x3 -> BN -> (+ skip) -> ReLU.

    The skip connection lets gradients flow straight through, which is *why*
    deep stacks of these train so much more easily than a plain conv tower.
    Spatial size is preserved (padding=1), so blocks can be stacked freely.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = F.relu(out + residual)  # the skip connection
        return out


class ConvTrunk(nn.Module):
    """Input conv + BN + ReLU, followed by ``num_blocks`` residual blocks.

    Returns a feature map of shape (batch, channels, H, W) -- spatial size
    unchanged from the input. Downstream heads decide how to pool/flatten it.
    """

    def __init__(self, in_channels: int, channels: int = 32, num_blocks: int = 3):
        super().__init__()
        self.input_conv = nn.Conv2d(in_channels, channels, kernel_size=3, padding=1, bias=False)
        self.input_bn = nn.BatchNorm2d(channels)
        self.blocks = nn.Sequential(*[ResidualBlock(channels) for _ in range(num_blocks)])
        self.channels = channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.input_bn(self.input_conv(x)))
        out = self.blocks(out)
        return out
