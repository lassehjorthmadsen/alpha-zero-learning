"""Foundation models for Track A (notebooks A1-A4).

Four architectures, each illustrating one idea you need for raccoon:

* :class:`MLP`          -- the plain baseline (A1).
* :class:`SmallResNet`  -- a CNN with residual blocks (A2); the spatial
                           inductive bias raccoon's trunk relies on.
* :class:`SeqRNN`       -- an LSTM/GRU over a sequence (A3); contrasts the
                           recurrent inductive bias with the CNN on the *same*
                           MNIST data fed row-by-row.
* :class:`TwoHeadedNet` -- one shared trunk, a classification head (softmax /
                           cross-entropy) and a regression head (tanh / MSE)
                           (A4). This is *structurally identical* to raccoon's
                           policy + value network -- see :class:`azl.network.AZNet`.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from azl.foundations.blocks import ConvTrunk


class MLP(nn.Module):
    """A plain fully-connected classifier over flattened inputs."""

    def __init__(self, in_features: int = 784, hidden: int = 256, num_classes: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SmallResNet(nn.Module):
    """A compact ResNet image classifier: ConvTrunk -> global-avg-pool -> linear.

    Defaults (32 channels, 3 blocks) reach >99% on MNIST in a few CPU epochs.
    """

    def __init__(
        self,
        in_channels: int = 1,
        channels: int = 32,
        num_blocks: int = 3,
        num_classes: int = 10,
    ):
        super().__init__()
        self.trunk = ConvTrunk(in_channels, channels, num_blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.trunk(x)               # (B, C, H, W)
        pooled = self.pool(feat).flatten(1)  # (B, C)
        return self.fc(pooled)             # (B, num_classes)


class SeqRNN(nn.Module):
    """An LSTM/GRU sequence classifier.

    For Sequential-MNIST each 28x28 image is read as 28 timesteps of 28 features
    (one row at a time). We classify from the final hidden state -- the same
    data as :class:`SmallResNet`, but a completely different inductive bias.
    """

    def __init__(
        self,
        input_size: int = 28,
        hidden_size: int = 128,
        num_layers: int = 1,
        num_classes: int = 10,
        cell: str = "lstm",
    ):
        super().__init__()
        cell = cell.lower()
        rnn_cls = {"lstm": nn.LSTM, "gru": nn.GRU, "rnn": nn.RNN}[cell]
        self.cell = cell
        self.rnn = rnn_cls(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, input_size)
        output, _ = self.rnn(x)
        last = output[:, -1, :]   # final-timestep hidden state
        return self.fc(last)


class TwoHeadedNet(nn.Module):
    """Shared trunk + classification head + regression head.

    forward(x) -> (class_logits, value)
        class_logits : (B, num_classes)  -- train with cross-entropy (like raccoon's POLICY head)
        value        : (B,)              -- tanh-bounded scalar in [-1, 1],
                                            train with MSE (like raccoon's VALUE head)

    Compare with :class:`azl.network.AZNet`: same skeleton, the only difference
    is the trunk's input shape and what the heads *mean* (move distribution vs
    digit class; position value vs digit magnitude).
    """

    def __init__(
        self,
        in_channels: int = 1,
        channels: int = 32,
        num_blocks: int = 3,
        num_classes: int = 10,
    ):
        super().__init__()
        self.trunk = ConvTrunk(in_channels, channels, num_blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.class_head = nn.Linear(channels, num_classes)
        self.value_head = nn.Linear(channels, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat = self.pool(self.trunk(x)).flatten(1)  # (B, C)
        class_logits = self.class_head(feat)
        value = torch.tanh(self.value_head(feat)).squeeze(-1)
        return class_logits, value
