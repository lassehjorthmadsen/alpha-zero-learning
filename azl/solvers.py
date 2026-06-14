"""Exact game solvers -- the teaching superpower of this repo.

A single negamax-with-chance routine (:class:`ExactSolver`) computes the *exact*
optimal value and optimal move(s) for any sufficiently small game implementing
the :class:`azl.games.base.GameState` interface. We use it on tic-tac-toe and
DiceRace (Connect-Four is far too large).

Why this matters: raccoon has no ground truth except GNUBG, so "is it actually
getting better?" is genuinely hard to answer. Here we *do* have ground truth, so
we can:

* **grade** a trained agent -- "optimal-move accuracy" and "does it ever lose?"
  (the headline verification in B2); and
* **generate labels** -- exact value/policy targets for the supervised
  pretraining lesson (B6), the toy analogue of raccoon's wildbg rollout data.

Values are returned from the *mover's* perspective, scaled to [-1, 1] by the
game's ``max_abs_return`` -- the same convention the network's value head uses.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from azl.games.base import GameState


class ExactSolver:
    """Memoised negamax + chance-expectation solver for a single game type."""

    def __init__(self):
        self._cache: dict = {}

    def value(self, state: GameState) -> float:
        """Optimal value from the current player's perspective, in [-1, 1]."""
        if state.is_terminal():
            return state.returns()[state.current_player()] / state.max_abs_return

        k = state.key()
        if k in self._cache:
            return self._cache[k]

        if state.is_chance():
            # Chance keeps the same player to move -> average child values directly.
            v = sum(p * self.value(state.apply_action(a)) for a, p in state.chance_outcomes())
        else:
            # Decision flips the player -> negate child values (negamax).
            v = max(-self.value(state.apply_action(a)) for a in state.legal_actions())

        self._cache[k] = v
        return v

    def action_values(self, state: GameState) -> dict[int, float]:
        """Map each legal action to the mover's value after playing it."""
        assert not state.is_chance() and not state.is_terminal()
        return {a: -self.value(state.apply_action(a)) for a in state.legal_actions()}

    def best_actions(self, state: GameState) -> list[int]:
        """All actions achieving the optimal value (ties included)."""
        av = self.action_values(state)
        best = max(av.values())
        return [a for a, v in av.items() if abs(v - best) < 1e-9]

    def policy_target(self, state: GameState) -> np.ndarray:
        """Uniform distribution over the optimal action(s), as a full-length vector."""
        best = self.best_actions(state)
        pi = np.zeros(state.num_actions, dtype=np.float32)
        for a in best:
            pi[a] = 1.0 / len(best)
        return pi


def enumerate_states(start: GameState, include_chance: bool = False, max_states: int = 1_000_000) -> list[GameState]:
    """Breadth-first enumeration of all reachable states from ``start``.

    Returns decision states by default (the ones a network is asked about);
    set ``include_chance=True`` to also include chance/terminal nodes.
    """
    seen: set = set()
    out: list[GameState] = []
    q: deque[GameState] = deque([start])
    seen.add(start.key())
    while q and len(seen) <= max_states:
        s = q.popleft()
        keep = include_chance or (not s.is_chance() and not s.is_terminal())
        if keep:
            out.append(s)
        if s.is_terminal():
            continue
        actions = (
            [a for a, _ in s.chance_outcomes()] if s.is_chance() else s.legal_actions()
        )
        for a in actions:
            child = s.apply_action(a)
            if child.key() not in seen:
                seen.add(child.key())
                q.append(child)
    return out


def solver_dataset(start: GameState, solver: ExactSolver | None = None):
    """Build (obs, value_targets, policy_targets) over all decision states.

    Used by B6 (supervised pretraining) and B5 (verifying the learned value
    matches ground truth).
    """
    solver = solver or ExactSolver()
    states = enumerate_states(start, include_chance=False)
    obs = np.stack([s.encode() for s in states])
    values = np.array([solver.value(s) for s in states], dtype=np.float32)
    policies = np.stack([solver.policy_target(s) for s in states])
    return obs, values, policies, states
