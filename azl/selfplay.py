"""Self-play game generation -- the data engine of AlphaZero.

``play_one_game`` plays a full game with MCTS choosing every move, recording for
each position: the board encoding, the **MCTS visit distribution** (the policy
target), and -- once the game ends -- a **value target**. This mirrors
``raccoon/train/self_play.py``, including:

* a **temperature schedule** (explore for the first ``temp_threshold`` moves,
  then play greedily) so early positions are diverse but later play is sharp; and
* **value bootstrapping** -- the value target blends the final game outcome with
  the MCTS root value via ``value_bootstrap_alpha`` (1.0 = pure outcome like the
  original AlphaZero, 0.0 = pure search value, lower variance).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from azl.games.base import GameState
from azl.mcts import MCTS, advance_through_chance, select_action
from azl.metrics import visit_entropy
from azl.network import AZNet


@dataclass
class TrainingExample:
    observation: np.ndarray   # encoded board, current player's perspective
    policy_target: np.ndarray  # length num_actions, sums to 1 (MCTS visit dist)
    value_target: float        # in [-1, 1], this position's player's perspective


@dataclass
class GameResult:
    examples: list[TrainingExample]
    num_moves: int
    returns: list[float]       # final [r0, r1]
    avg_visit_entropy: float


def play_one_game(
    net: AZNet,
    game_cls,
    num_simulations: int = 50,
    c_puct: float = 1.5,
    temperature: float = 1.0,
    temp_threshold: int = 4,
    dirichlet_alpha: float = 0.0,
    noise_eps: float = 0.25,
    value_bootstrap_alpha: float = 1.0,
    rng: np.random.Generator | None = None,
) -> GameResult:
    rng = rng or np.random.default_rng()
    mcts = MCTS(net, num_simulations, c_puct, dirichlet_alpha, noise_eps, rng)

    state: GameState = game_cls.start()
    state = advance_through_chance(state, rng)   # the real opening roll, if any

    history: list[tuple[np.ndarray, np.ndarray, int, float]] = []
    entropies: list[float] = []
    move = 0
    while not state.is_terminal():
        visit_probs, root_q, visit_counts = mcts.search(state)
        if not visit_counts:
            break

        policy = np.zeros(game_cls.num_actions, dtype=np.float32)
        for a, p in visit_probs.items():
            policy[a] = p
        history.append((state.encode(), policy, state.current_player(), root_q))
        entropies.append(visit_entropy(visit_probs))

        temp = temperature if move < temp_threshold else 0.0
        action = select_action(visit_counts, temperature=temp, rng=rng)
        state = state.apply_action(action)
        state = advance_through_chance(state, rng)
        move += 1

    returns = state.returns()
    max_r = game_cls.max_abs_return
    examples = []
    for obs, policy, player, root_q in history:
        outcome = returns[player] / max_r
        value_target = float(
            np.clip(value_bootstrap_alpha * outcome + (1 - value_bootstrap_alpha) * root_q, -1.0, 1.0)
        )
        examples.append(TrainingExample(obs, policy, value_target))

    return GameResult(
        examples=examples,
        num_moves=move,
        returns=returns,
        avg_visit_entropy=float(np.mean(entropies)) if entropies else 0.0,
    )
