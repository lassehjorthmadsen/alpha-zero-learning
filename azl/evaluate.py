"""Players and head-to-head matches -- the toy analogue of raccoon's arena/GNUBG.

A *player* is just a function ``state -> action``. We provide:

* :func:`random_player`   -- uniform legal move (the weakest baseline);
* :func:`solver_player`   -- plays an exact optimal move (perfect play);
* :func:`net_player`      -- the raw network policy, no search;
* :func:`mcts_player`     -- network + MCTS (how the agent really plays).

:func:`play_match` pits two players over many games (alternating who starts) and
reports the result from player A's perspective. Because tic-tac-toe/DiceRace are
solved, pitting ``mcts_player`` against ``solver_player`` is a *hard* test:
optimal play can only be drawn, so "A never loses to the solver" means A has
essentially learned the game (the headline check in B2).
"""

from __future__ import annotations

import numpy as np

from azl.mcts import MCTS, advance_through_chance, select_action
from azl.metrics import policy_greedy_action


def random_player(rng: np.random.Generator):
    def play(state):
        return int(rng.choice(state.legal_actions()))
    return play


def solver_player(solver, rng: np.random.Generator):
    def play(state):
        best = solver.best_actions(state)
        return int(rng.choice(best))
    return play


def net_player(net):
    def play(state):
        return policy_greedy_action(net, state)
    return play


def mcts_player(net, num_simulations: int = 100, temperature: float = 0.0, rng: np.random.Generator | None = None):
    rng = rng or np.random.default_rng()
    mcts = MCTS(net, num_simulations=num_simulations, rng=rng)
    def play(state):
        _, _, visit_counts = mcts.search(state, add_noise=False)
        return select_action(visit_counts, temperature=temperature, rng=rng)
    return play


def play_match(game_cls, player_a, player_b, num_games: int = 50, rng=None, alternate: bool = True) -> dict:
    """Play ``num_games`` and return results from player A's perspective."""
    rng = rng or np.random.default_rng()
    a_wins = a_losses = draws = 0
    a_return_total = 0.0

    for g in range(num_games):
        a_is_p0 = (g % 2 == 0) if alternate else True
        state = advance_through_chance(game_cls.start(), rng)
        while not state.is_terminal():
            a_to_move = (state.current_player() == 0) == a_is_p0
            action = (player_a if a_to_move else player_b)(state)
            state = state.apply_action(action)
            state = advance_through_chance(state, rng)

        r0 = state.returns()[0]
        a_return = r0 if a_is_p0 else -r0
        a_return_total += a_return
        if a_return > 0:
            a_wins += 1
        elif a_return < 0:
            a_losses += 1
        else:
            draws += 1

    return {
        "num_games": num_games,
        "a_wins": a_wins,
        "draws": draws,
        "a_losses": a_losses,
        "a_return_mean": a_return_total / num_games,
    }
