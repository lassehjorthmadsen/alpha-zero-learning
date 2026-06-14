"""The minimal game interface every toy game implements.

This is a deliberately small subset of raccoon's ``GameState`` (which wraps
OpenSpiel). The methods below are everything the MCTS, self-play and solver code
needs. Key conventions:

* **Two players, 0 and 1, zero-sum.** ``returns()`` gives ``[r0, r1]`` with
  ``r1 == -r0``.
* **Immutable states.** ``apply_action`` returns a *new* state; it never mutates
  ``self``. (Raccoon clones-then-mutates for speed; we favour clarity. The
  difference is explained in ``notebooks/B7_map_to_raccoon.ipynb``.)
* **Current player's perspective.** ``encode()`` always describes the position
  from the mover's point of view -- "my checkers / their checkers" -- so the
  network never needs to know an absolute side. Same idea as raccoon's encoder.
* **Chance nodes.** Deterministic games leave ``is_chance()`` False. A
  stochastic game (``dice_race``) returns True at a dice roll; the MCTS samples
  ``chance_outcomes()`` and skips such nodes, exactly like raccoon.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class GameState(ABC):
    # Size of the (fixed) action space -- the policy head outputs this many logits.
    num_actions: int
    # Largest possible |return|; value targets are scaled to [-1, 1] by dividing
    # by this (1 for win/lose games, 2 for dice_race where a "gammon" is worth 2).
    max_abs_return: float = 1.0

    # --- core (every game must implement) ---------------------------------
    @abstractmethod
    def current_player(self) -> int:
        """Index (0 or 1) of the player to move."""

    @abstractmethod
    def legal_actions(self) -> list[int]:
        """Legal action indices at a *decision* node."""

    @abstractmethod
    def apply_action(self, action: int) -> "GameState":
        """Return the new state after ``action`` (no mutation of self)."""

    @abstractmethod
    def is_terminal(self) -> bool: ...

    @abstractmethod
    def returns(self) -> list[float]:
        """Final scores ``[r0, r1]`` (only meaningful when terminal)."""

    @abstractmethod
    def encode(self) -> np.ndarray:
        """Network input tensor from the current player's perspective."""

    @abstractmethod
    def key(self):
        """A hashable key uniquely identifying this state (for solver memoisation)."""

    # --- chance support (default: deterministic game) ---------------------
    def is_chance(self) -> bool:
        return False

    def chance_outcomes(self) -> list[tuple[int, float]]:
        """``[(action, probability), ...]`` at a chance node; empty otherwise."""
        return []

    # --- convenience ------------------------------------------------------
    @property
    def encoded_shape(self) -> tuple[int, ...]:
        return tuple(self.encode().shape)

    def render(self) -> str:
        return repr(self)
