"""Tic-tac-toe: the smallest fully-solvable testbed for AlphaZero.

Deterministic, 9 actions (cell index 0-8), ~5478 reachable states. Because it is
trivially solvable by minimax (see :mod:`azl.solvers`), we can *grade* a trained
AlphaZero agent against ground truth -- the verifiable check at the heart of B2.
"""

from __future__ import annotations

import numpy as np

from azl.games.base import GameState

# The 8 winning lines (cell indices).
_LINES = [
    (0, 1, 2), (3, 4, 5), (6, 7, 8),   # rows
    (0, 3, 6), (1, 4, 7), (2, 5, 8),   # cols
    (0, 4, 8), (2, 4, 6),              # diagonals
]


class TicTacToe(GameState):
    num_actions = 9
    max_abs_return = 1.0

    def __init__(self, board: tuple[int, ...] | None = None, player: int = 0):
        # board cell values: 0 empty, 1 = player 0's mark, 2 = player 1's mark.
        self.board = board if board is not None else (0,) * 9
        self.player = player

    @classmethod
    def start(cls) -> "TicTacToe":
        return cls()

    def current_player(self) -> int:
        return self.player

    def _winner(self) -> int | None:
        for a, b, c in _LINES:
            v = self.board[a]
            if v != 0 and v == self.board[b] == self.board[c]:
                return v - 1  # marker 1 -> player 0, marker 2 -> player 1
        return None

    def is_terminal(self) -> bool:
        return self._winner() is not None or all(v != 0 for v in self.board)

    def legal_actions(self) -> list[int]:
        if self.is_terminal():
            return []
        return [i for i, v in enumerate(self.board) if v == 0]

    def apply_action(self, action: int) -> "TicTacToe":
        if self.board[action] != 0:
            raise ValueError(f"cell {action} is occupied")
        new_board = list(self.board)
        new_board[action] = self.player + 1
        return TicTacToe(tuple(new_board), 1 - self.player)

    def returns(self) -> list[float]:
        w = self._winner()
        if w is None:
            return [0.0, 0.0]
        return [1.0, -1.0] if w == 0 else [-1.0, 1.0]

    def encode(self) -> np.ndarray:
        """(2, 3, 3): plane 0 = my marks, plane 1 = opponent marks."""
        me = self.player + 1
        t = np.zeros((2, 3, 3), dtype=np.float32)
        for i, v in enumerate(self.board):
            if v == me:
                t[0, i // 3, i % 3] = 1.0
            elif v != 0:
                t[1, i // 3, i % 3] = 1.0
        return t

    def key(self):
        return (self.board, self.player)

    def render(self) -> str:
        sym = {0: ".", 1: "X", 2: "O"}
        rows = [" ".join(sym[self.board[r * 3 + c]] for c in range(3)) for r in range(3)]
        return "\n".join(rows)
