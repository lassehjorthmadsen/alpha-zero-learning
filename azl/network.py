"""AZNet -- the two-headed policy/value network for the AlphaZero track.

This is the toy twin of raccoon's ``RaccoonNet`` (``raccoon/model/network.py``):
a shared trunk feeding a **policy head** (one logit per action) and a **value
head** (a single tanh-bounded scalar in [-1, 1]). The only real differences are
size and the choice of trunk:

* ``trunk="mlp"``    -- a small fully-connected trunk, for the flat/tiny inputs
  of tic-tac-toe and DiceRace.
* ``trunk="resnet"`` -- the :class:`azl.foundations.blocks.ConvTrunk` (the very
  residual blocks you trained on MNIST in A2), for the spatial Connect-Four
  board. This is the same conv-trunk + policy/value-head design raccoon uses.

``predict`` is the single-position inference used by the MCTS: it masks illegal
actions to -inf before softmax, exactly like raccoon.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from azl.foundations.blocks import ConvTrunk


class AZNet(nn.Module):
    def __init__(
        self,
        input_shape: tuple[int, ...],
        num_actions: int,
        trunk: str = "mlp",
        hidden: int = 128,
        channels: int = 32,
        num_blocks: int = 3,
    ):
        super().__init__()
        self.input_shape = tuple(input_shape)
        self.num_actions = num_actions
        self.trunk_kind = trunk
        # Stored so checkpoints can rebuild the exact architecture (like raccoon).
        self.config = {
            "input_shape": self.input_shape,
            "num_actions": num_actions,
            "trunk": trunk,
            "hidden": hidden,
            "channels": channels,
            "num_blocks": num_blocks,
        }

        if trunk == "mlp":
            in_dim = int(math.prod(self.input_shape))
            self.trunk = nn.Sequential(
                nn.Flatten(),
                nn.Linear(in_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
            )
            self.policy_head = nn.Linear(hidden, num_actions)
            self.value_head = nn.Sequential(
                nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Linear(hidden // 2, 1)
            )
        elif trunk == "resnet":
            c, h, w = self.input_shape
            self.trunk = ConvTrunk(in_channels=c, channels=channels, num_blocks=num_blocks)
            # Policy head: 1x1 conv -> flatten -> linear  (mirrors raccoon)
            self.policy_conv = nn.Conv2d(channels, 2, kernel_size=1)
            self.policy_bn = nn.BatchNorm2d(2)
            self.policy_fc = nn.Linear(2 * h * w, num_actions)
            # Value head: 1x1 conv -> flatten -> linear -> linear
            self.value_conv = nn.Conv2d(channels, 1, kernel_size=1)
            self.value_bn = nn.BatchNorm2d(1)
            self.value_fc1 = nn.Linear(h * w, hidden)
            self.value_fc2 = nn.Linear(hidden, 1)
        else:
            raise ValueError(f"unknown trunk {trunk!r}; use 'mlp' or 'resnet'")

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.trunk_kind == "mlp":
            feat = self.trunk(x)
            policy_logits = self.policy_head(feat)
            value = torch.tanh(self.value_head(feat)).squeeze(-1)
            return policy_logits, value

        # resnet
        feat = self.trunk(x)                                   # (B, C, H, W)
        p = F.relu(self.policy_bn(self.policy_conv(feat)))
        policy_logits = self.policy_fc(p.flatten(1))
        v = F.relu(self.value_bn(self.value_conv(feat)))
        v = F.relu(self.value_fc1(v.flatten(1)))
        value = torch.tanh(self.value_fc2(v)).squeeze(-1)
        return policy_logits, value

    @torch.no_grad()
    def predict(self, obs: np.ndarray, legal_actions: list[int]) -> tuple[dict[int, float], float]:
        """Single-position inference for MCTS.

        Returns ``(policy, value)`` where ``policy`` maps each legal action to a
        probability (illegal actions masked out before softmax) and ``value`` is
        the scalar value in [-1, 1] from the current player's perspective.
        """
        was_training = self.training
        self.eval()
        x = torch.from_numpy(np.asarray(obs)).unsqueeze(0).float().to(self.device)
        logits, value = self.forward(x)
        logits = logits.squeeze(0).cpu()

        mask = torch.full((self.num_actions,), float("-inf"))
        mask[legal_actions] = 0.0
        probs = F.softmax(logits + mask, dim=0).numpy()

        if was_training:
            self.train()
        policy = {a: float(probs[a]) for a in legal_actions}
        return policy, float(value.item())


def save_checkpoint(net: AZNet, path: str) -> None:
    torch.save({"config": net.config, "state_dict": net.state_dict()}, path)


def load_checkpoint(path: str, map_location="cpu") -> AZNet:
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    net = AZNet(**ckpt["config"])
    net.load_state_dict(ckpt["state_dict"])
    return net


def net_for_game(game_cls, trunk: str | None = None, **kwargs) -> AZNet:
    """Construct an :class:`AZNet` sized for a game class.

    Picks a sensible default trunk: ``resnet`` for the spatial Connect-Four
    board, ``mlp`` otherwise. ``game_cls.start().encode().shape`` gives the
    input shape.
    """
    start = game_cls.start()
    input_shape = start.encode().shape
    if trunk is None:
        trunk = "resnet" if len(input_shape) == 3 and input_shape[1] >= 5 else "mlp"
    return AZNet(input_shape=input_shape, num_actions=game_cls.num_actions, trunk=trunk, **kwargs)
