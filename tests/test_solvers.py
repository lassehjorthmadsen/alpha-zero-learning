"""Exact-solver tests -- these double as the ground truth used to grade agents."""

from azl.games.tictactoe import TicTacToe
from azl.games.dice_race import DiceRace
from azl.solvers import ExactSolver, enumerate_states, solver_dataset


def test_ttt_optimal_play_is_a_draw():
    # The famous result: tic-tac-toe is a draw under perfect play.
    solver = ExactSolver()
    assert abs(solver.value(TicTacToe.start())) < 1e-9


def test_ttt_solver_takes_an_immediate_win():
    # X about to complete the top row at cell 2; that is the unique winning move.
    s = TicTacToe(board=(1, 1, 0, 2, 2, 0, 0, 0, 0), player=0)
    solver = ExactSolver()
    assert solver.value(s) == 1.0
    assert solver.best_actions(s) == [2]


def test_ttt_enumeration_count():
    # Reachable decision states (non-terminal) -- a stable sanity number.
    states = enumerate_states(TicTacToe.start(), include_chance=False)
    assert 4000 < len(states) < 6000


def test_dicerace_value_is_bounded_and_solvable():
    solver = ExactSolver()
    v = solver.value(DiceRace.start())     # start is a chance node
    assert -1.0 <= v <= 1.0


def test_solver_dataset_shapes():
    obs, values, policies, states = solver_dataset(DiceRace.start())
    assert obs.shape[0] == values.shape[0] == policies.shape[0] == len(states)
    assert policies.shape[1] == DiceRace.num_actions
    # every policy target is a probability distribution
    assert abs(policies.sum(axis=1).mean() - 1.0) < 1e-6
