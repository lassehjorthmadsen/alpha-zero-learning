"""DiceRace -- a tiny stochastic race that introduces *chance nodes*.

This is the backgammon bridge. Two players race ``N_CHECKERS`` checkers home;
"home" means a checker's *pip distance* reaches 0 (it bears off). Each turn:

1. **Chance node** -- roll one die in ``1..DIE_FACES`` (``is_chance()`` is True;
   the MCTS samples and skips it, never asking the network -- exactly raccoon's
   ``_advance_through_chance``).
2. **Decision node** -- choose *which* checker to advance by the rolled value. A
   checker at distance ``d`` moves to ``d - die``, or bears off if ``d <= die``.

First to bear off all checkers wins (+1); if the loser has borne off none it is a
"gammon" worth +2 (``max_abs_return == 2``) -- a graded outcome like backgammon's
gammons/backgammons, so the value target is richer than win/lose.

Why so small (2 checkers, start pip 6, 3-sided die)? So the whole game is
**exactly solvable** (:class:`azl.solvers.ExactSolver`), letting B5/B6 compare the
learned value to ground truth. Coordinates are *distance to home*, which is
already player-relative, so the perspective flip in ``encode`` is just swapping
"my" and "their" checkers -- no board mirroring needed.

There is deliberately **no hitting or blocking** here (the two races are
independent); that keeps the state space a DAG and the solver exact. Real
backgammon's contact/hitting is what makes raccoon hard -- noted in B5.
"""

from __future__ import annotations

import numpy as np

from azl.games.base import GameState

START_PIP = 6
N_CHECKERS = 2
DIE_FACES = 3


class DiceRace(GameState):
    num_actions = START_PIP + 1          # action == the pip-distance of the checker to move (0 unused)
    max_abs_return = 2.0                  # gammon is worth 2

    START_PIP = START_PIP
    N_CHECKERS = N_CHECKERS
    DIE_FACES = DIE_FACES

    def __init__(
        self,
        pips0: tuple[int, ...] | None = None,
        pips1: tuple[int, ...] | None = None,
        player: int = 0,
        die: int | None = None,
    ):
        # pipsX = sorted tuple of remaining checker distances (those not borne off).
        self.pips0 = pips0 if pips0 is not None else (START_PIP,) * N_CHECKERS
        self.pips1 = pips1 if pips1 is not None else (START_PIP,) * N_CHECKERS
        self.player = player
        self.die = die  # None => a roll is pending (chance node)

    @classmethod
    def start(cls) -> "DiceRace":
        # Player 0 to roll first.
        return cls(player=0, die=None)

    # --- helpers ----------------------------------------------------------
    def _my(self) -> tuple[int, ...]:
        return self.pips0 if self.player == 0 else self.pips1

    def _opp(self) -> tuple[int, ...]:
        return self.pips1 if self.player == 0 else self.pips0

    def current_player(self) -> int:
        return self.player

    def is_terminal(self) -> bool:
        return len(self.pips0) == 0 or len(self.pips1) == 0

    # --- chance node ------------------------------------------------------
    def is_chance(self) -> bool:
        return self.die is None and not self.is_terminal()

    def chance_outcomes(self) -> list[tuple[int, float]]:
        p = 1.0 / DIE_FACES
        return [(face, p) for face in range(1, DIE_FACES + 1)]

    # --- decision node ----------------------------------------------------
    def legal_actions(self) -> list[int]:
        if self.die is None or self.is_terminal():
            return []
        # Every checker can act with a single die (move or bear off). The action
        # is the distinct distance to advance one checker from.
        return sorted(set(self._my()))

    def apply_action(self, action: int) -> "DiceRace":
        if self.is_chance():
            # 'action' is the rolled die face.
            return DiceRace(self.pips0, self.pips1, self.player, die=action)

        # Decision: advance one checker currently at distance `action`.
        mine = list(self._my())
        mine.remove(action)                 # raises if illegal -- a useful guard
        if action > self.die:
            mine.append(action - self.die)  # plain move
        # else: action <= die -> the checker bears off (drops out of the list)
        mine = tuple(sorted(mine))

        if self.player == 0:
            new0, new1 = mine, self.pips1
        else:
            new0, new1 = self.pips0, mine
        # Hand over to the opponent with a fresh roll pending.
        return DiceRace(new0, new1, 1 - self.player, die=None)

    def returns(self) -> list[float]:
        if not self.is_terminal():
            return [0.0, 0.0]
        winner = 0 if len(self.pips0) == 0 else 1
        loser_pips = self.pips1 if winner == 0 else self.pips0
        gammon = len(loser_pips) == N_CHECKERS      # loser borne off nothing
        v = 2.0 if gammon else 1.0
        return [v, -v] if winner == 0 else [-v, v]

    def encode(self) -> np.ndarray:
        """1-D feature vector from the mover's perspective.

        Layout (length 2*START_PIP + 3):
            my checker counts at distances 1..START_PIP, my borne-off count,
            opp checker counts at distances 1..START_PIP, opp borne-off count,
            normalised die (die/DIE_FACES, or 0 at a chance node).
        """
        feat = np.zeros(2 * START_PIP + 3, dtype=np.float32)
        my, opp = self._my(), self._opp()
        for d in my:
            feat[d - 1] += 1.0
        feat[START_PIP] = N_CHECKERS - len(my)            # my borne-off count
        for d in opp:
            feat[START_PIP + 1 + (d - 1)] += 1.0
        feat[2 * START_PIP + 1] = N_CHECKERS - len(opp)   # opp borne-off count
        feat[2 * START_PIP + 2] = (self.die or 0) / DIE_FACES
        return feat

    def key(self):
        return (self.pips0, self.pips1, self.player, self.die)

    def render(self) -> str:
        return (
            f"P0 pips={self.pips0} off={N_CHECKERS - len(self.pips0)} | "
            f"P1 pips={self.pips1} off={N_CHECKERS - len(self.pips1)} | "
            f"to_move=P{self.player} die={self.die}"
        )
