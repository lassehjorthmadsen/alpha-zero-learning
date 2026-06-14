"""Toy games for the AlphaZero track (Track B), written from scratch.

Each game implements the small :class:`azl.games.base.GameState` protocol:
deterministic games (tic-tac-toe, Connect-Four) plus one stochastic game
(``dice_race``) whose *chance nodes* mirror backgammon's dice rolls in raccoon.
"""

from azl.games.base import GameState  # noqa: F401
from azl.games.tictactoe import TicTacToe
from azl.games.connect4 import Connect4
from azl.games.dice_race import DiceRace

# Name -> game class, for the CLI scripts and notebooks.
GAMES: dict[str, type[GameState]] = {
    "tictactoe": TicTacToe,
    "connect4": Connect4,
    "dice_race": DiceRace,
}


def get_game(name: str) -> type[GameState]:
    try:
        return GAMES[name]
    except KeyError:
        raise SystemExit(f"unknown game {name!r}; choose from {sorted(GAMES)}")

