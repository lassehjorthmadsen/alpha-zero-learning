"""The AlphaZero training loop -- self-play, train, repeat (raccoon's ``Coach``).

One ``iteration`` is:

1. **Self-play** -- play ``games_per_iter`` games with the current network; push
   every (obs, MCTS-policy, value) example into the replay buffer.
2. **Train** -- take ``training_steps`` SGD steps on random batches, minimising

       loss = cross_entropy(policy_logits, mcts_policy)   # the POLICY loss
            + mse(value, value_target)                    # the VALUE loss

   (weight decay supplies the L2 term). This is exactly raccoon's objective.
3. **Log** -- record losses and game statistics; optionally call a user callback
   so notebooks can also measure solver accuracy, arena strength, etc.

The network that generated the data is the same one being trained, so the data
distribution keeps shifting underneath you -- the *non-stationarity* that makes
AlphaZero training dynamics so interesting (and is the subject of B3).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F

from azl.network import AZNet
from azl.replay_buffer import ReplayBuffer
from azl.selfplay import play_one_game


@dataclass
class CoachConfig:
    games_per_iter: int = 20
    num_simulations: int = 50
    c_puct: float = 1.5
    temperature: float = 1.0
    temp_threshold: int = 4
    dirichlet_alpha: float = 0.0
    noise_eps: float = 0.25
    value_bootstrap_alpha: float = 1.0
    replay_size: int = 50_000
    batch_size: int = 128
    training_steps: int = 40
    lr: float = 1e-3
    weight_decay: float = 1e-4


@dataclass
class Coach:
    net: AZNet
    game_cls: type
    config: CoachConfig = field(default_factory=CoachConfig)
    device: str = "cpu"
    seed: int = 0
    log_path: str | None = None

    def __post_init__(self):
        self.net.to(self.device)
        self.rng = np.random.default_rng(self.seed)
        self.replay = ReplayBuffer(self.config.replay_size)
        self.optimizer = torch.optim.Adam(
            self.net.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay
        )
        self.history: list[dict] = []
        if self.log_path:
            with open(self.log_path, "w") as f:
                f.write(json.dumps({"config": self.config.__dict__, "game": self.game_cls.__name__}) + "\n")

    # --- phases -----------------------------------------------------------
    def self_play_phase(self) -> list:
        cfg = self.config
        results = []
        for _ in range(cfg.games_per_iter):
            res = play_one_game(
                self.net, self.game_cls,
                num_simulations=cfg.num_simulations, c_puct=cfg.c_puct,
                temperature=cfg.temperature, temp_threshold=cfg.temp_threshold,
                dirichlet_alpha=cfg.dirichlet_alpha, noise_eps=cfg.noise_eps,
                value_bootstrap_alpha=cfg.value_bootstrap_alpha, rng=self.rng,
            )
            self.replay.add_game(res.examples)
            results.append(res)
        return results

    def train_phase(self) -> dict:
        cfg = self.config
        self.net.train()
        tot_p, tot_v, steps = 0.0, 0.0, 0
        for _ in range(cfg.training_steps):
            obs, target_policy, target_value = self.replay.sample_batch(cfg.batch_size)
            obs = obs.to(self.device)
            target_policy = target_policy.to(self.device)
            target_value = target_value.to(self.device)

            logits, value = self.net(obs)
            policy_loss = -(target_policy * F.log_softmax(logits, dim=1)).sum(dim=1).mean()
            value_loss = F.mse_loss(value, target_value)
            loss = policy_loss + value_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            tot_p += policy_loss.item()
            tot_v += value_loss.item()
            steps += 1
        return {
            "policy_loss": tot_p / max(steps, 1),
            "value_loss": tot_v / max(steps, 1),
        }

    # --- driver -----------------------------------------------------------
    def run_iteration(self, iteration: int) -> dict:
        t0 = time.time()
        results = self.self_play_phase()
        metrics = (
            self.train_phase()
            if len(self.replay) >= self.config.batch_size
            else {"policy_loss": float("nan"), "value_loss": float("nan")}
        )

        moves = [r.num_moves for r in results]
        ent = [r.avg_visit_entropy for r in results]
        p0 = [r.returns[0] for r in results]
        metrics.update(
            iteration=iteration,
            buffer=len(self.replay),
            avg_moves=float(np.mean(moves)),
            avg_visit_entropy=float(np.mean(ent)),
            p0_return=float(np.mean(p0)),
            seconds=round(time.time() - t0, 2),
        )
        self.history.append(metrics)
        if self.log_path:
            with open(self.log_path, "a") as f:
                f.write(json.dumps(metrics) + "\n")
        return metrics

    def train(
        self,
        iterations: int,
        callback: Callable[[int, dict, "Coach"], None] | None = None,
        verbose: bool = True,
    ) -> list[dict]:
        for i in range(1, iterations + 1):
            m = self.run_iteration(i)
            if callback is not None:
                callback(i, m, self)
            if verbose:
                print(
                    f"iter {i:>3}  policy {m['policy_loss']:.3f}  value {m['value_loss']:.3f}  "
                    f"entropy {m['avg_visit_entropy']:.3f}  moves {m['avg_moves']:.1f}  "
                    f"p0_ret {m['p0_return']:+.2f}  ({m['seconds']}s)"
                )
        return self.history
