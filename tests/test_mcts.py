"""MCTS tests: visit distribution validity, winning-move preference, chance nodes."""

import numpy as np
import torch

from azl.games.tictactoe import TicTacToe
from azl.games.dice_race import DiceRace
from azl.mcts import MCTS, select_action
from azl.network import net_for_game


def test_visit_distribution_is_valid():
    torch.manual_seed(0)
    net = net_for_game(TicTacToe)
    mcts = MCTS(net, num_simulations=40, rng=np.random.default_rng(0))
    visit_probs, root_value, visit_counts = mcts.search(TicTacToe.start())
    assert abs(sum(visit_probs.values()) - 1.0) < 1e-9
    assert sum(visit_counts.values()) == 40
    assert set(visit_probs) <= set(TicTacToe.start().legal_actions())
    assert -1.0 <= root_value <= 1.0


def test_mcts_finds_immediate_win():
    # Player 0 (X) can win by playing cell 2 (completing the top row).
    s = TicTacToe(board=(1, 1, 0, 2, 2, 0, 0, 0, 0), player=0)
    torch.manual_seed(0)
    net = net_for_game(TicTacToe)
    mcts = MCTS(net, num_simulations=150, rng=np.random.default_rng(0))
    _, _, visit_counts = mcts.search(s)
    best = select_action(visit_counts, temperature=0.0)
    assert best == 2, f"MCTS should take the winning move, got {best}"


def test_mcts_handles_chance_nodes():
    # DiceRace search must run through chance (dice) nodes without error.
    from azl.mcts import advance_through_chance

    rng = np.random.default_rng(0)
    torch.manual_seed(0)
    net = net_for_game(DiceRace)
    state = advance_through_chance(DiceRace.start(), rng)   # real opening roll -> decision node
    mcts = MCTS(net, num_simulations=60, rng=rng)
    visit_probs, root_value, visit_counts = mcts.search(state)
    assert sum(visit_counts.values()) == 60
    assert set(visit_probs) <= set(state.legal_actions())
