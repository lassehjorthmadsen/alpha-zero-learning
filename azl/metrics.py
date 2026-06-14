"""Diagnostics for watching training -- and *grading* it against the solver.

Two kinds of metric:

* **exploration health** -- ``visit_entropy`` / ``effective_moves`` measure how
  spread-out the MCTS visit distribution is. Collapsing entropy is the
  fingerprint of the exploration-collapse plateau raccoon documented (B3).
* **ground-truth grading** -- because tic-tac-toe and DiceRace are solved, we can
  measure ``optimal_action_accuracy`` (how often the agent picks an optimal move)
  and ``value_mae_vs_solver`` (how far the value head is from the true value).
  This is the verifiable progress signal raccoon can only approximate via GNUBG.
"""

from __future__ import annotations

import math
from typing import Callable

import numpy as np

from azl.games.base import GameState


def visit_entropy(visit_probs: dict[int, float]) -> float:
    """Shannon entropy (nats) of an MCTS visit distribution."""
    return -sum(p * math.log(p) for p in visit_probs.values() if p > 0.0)


def effective_moves(entropy: float) -> float:
    """exp(H): the "effective number of moves" being explored."""
    return math.exp(entropy)


def policy_greedy_action(net, state: GameState) -> int:
    """The move the *raw network policy* prefers (no search)."""
    policy, _ = net.predict(state.encode(), state.legal_actions())
    return max(policy, key=policy.get)


def optimal_action_accuracy(action_fn: Callable[[GameState], int], states, solver) -> float:
    """Fraction of ``states`` where ``action_fn`` returns an optimal move."""
    if not states:
        return float("nan")
    hits = sum(action_fn(s) in solver.best_actions(s) for s in states)
    return hits / len(states)


def value_mae_vs_solver(net, states, solver) -> float:
    """Mean absolute error between the value head and the exact solver value."""
    if not states:
        return float("nan")
    errs = [abs(net.predict(s.encode(), s.legal_actions())[1] - solver.value(s)) for s in states]
    return float(np.mean(errs))
