"""Connect-Four: a bigger, spatial, still-deterministic game.

6 rows x 7 columns, 7 actions (the column to drop into). Too large to solve by
brute force, so here we judge strength by self-play arena and by play against
simple baselines -- but its 2D structure is exactly why a *ResNet* trunk
(``trunk="resnet"`` in :class:`azl.network.AZNet`) helps, mirroring raccoon.
"""

from __future__ import annotations

import numpy as np

from azl.games.base import GameState

ROWS, COLS = 6, 7


def _idx(r: int, c: int) -> int:
    return r * COLS + c


class Connect4(GameState):
    num_actions = COLS
    max_abs_return = 1.0

    def __init__(self, board: tuple[int, ...] | None = None, player: int = 0, last: int | None = None):
        # 42 cells, row 0 = top, row 5 = bottom. Values 0 empty / 1 player0 / 2 player1.
        self.board = board if board is not None else (0,) * (ROWS * COLS)
        self.player = player
        self._last = last  # (r, c) of the most recent drop, for cheap win checks

    @classmethod
    def start(cls) -> "Connect4":
        return cls()

    def current_player(self) -> int:
        return self.player

    def _landing_row(self, col: int) -> int | None:
        for r in range(ROWS - 1, -1, -1):  # bottom-up
            if self.board[_idx(r, col)] == 0:
                return r
        return None

    def legal_actions(self) -> list[int]:
        if self.is_terminal():
            return []
        return [c for c in range(COLS) if self.board[_idx(0, c)] == 0]

    def apply_action(self, action: int) -> "Connect4":
        r = self._landing_row(action)
        if r is None:
            raise ValueError(f"column {action} is full")
        new_board = list(self.board)
        new_board[_idx(r, action)] = self.player + 1
        return Connect4(tuple(new_board), 1 - self.player, last=(r, action))

    def _winner(self) -> int | None:
        # Only the last move can complete a line; if unknown, scan everything.
        if self._last is None:
            cells = [(r, c) for r in range(ROWS) for c in range(COLS)]
        else:
            cells = [self._last]
        for r0, c0 in cells:
            v = self.board[_idx(r0, c0)]
            if v == 0:
                continue
            for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
                count = 1
                for sign in (1, -1):
                    r, c = r0 + dr * sign, c0 + dc * sign
                    while 0 <= r < ROWS and 0 <= c < COLS and self.board[_idx(r, c)] == v:
                        count += 1
                        r += dr * sign
                        c += dc * sign
                if count >= 4:
                    return v - 1
        return None

    def is_terminal(self) -> bool:
        return self._winner() is not None or all(self.board[_idx(0, c)] != 0 for c in range(COLS))

    def returns(self) -> list[float]:
        w = self._winner()
        if w is None:
            return [0.0, 0.0]
        return [1.0, -1.0] if w == 0 else [-1.0, 1.0]

    def encode(self) -> np.ndarray:
        """(2, 6, 7): plane 0 = my discs, plane 1 = opponent discs."""
        me = self.player + 1
        t = np.zeros((2, ROWS, COLS), dtype=np.float32)
        for r in range(ROWS):
            for c in range(COLS):
                v = self.board[_idx(r, c)]
                if v == me:
                    t[0, r, c] = 1.0
                elif v != 0:
                    t[1, r, c] = 1.0
        return t

    def key(self):
        return (self.board, self.player)

    def render(self) -> str:
        sym = {0: ".", 1: "X", 2: "O"}
        rows = [" ".join(sym[self.board[_idx(r, c)]] for c in range(COLS)) for r in range(ROWS)]
        footer = " ".join(str(c) for c in range(COLS))
        return "\n".join(rows) + "\n" + footer
