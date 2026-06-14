"""Self-play + replay-buffer tests."""

import numpy as np
import torch

from azl.games.tictactoe import TicTacToe
from azl.network import net_for_game
from azl.replay_buffer import ReplayBuffer
from azl.selfplay import play_one_game


def test_self_play_produces_consistent_examples():
    torch.manual_seed(0)
    net = net_for_game(TicTacToe)
    result = play_one_game(net, TicTacToe, num_simulations=20, rng=np.random.default_rng(0))
    assert result.num_moves >= 5
    assert result.returns[0] == -result.returns[1]      # zero-sum
    for ex in result.examples:
        assert ex.observation.shape == (2, 3, 3)
        assert ex.policy_target.shape == (9,)
        assert abs(ex.policy_target.sum() - 1.0) < 1e-5
        assert -1.0 <= ex.value_target <= 1.0


def test_replay_buffer_add_and_sample():
    torch.manual_seed(0)
    net = net_for_game(TicTacToe)
    buf = ReplayBuffer(max_size=500)
    for seed in range(3):
        result = play_one_game(net, TicTacToe, num_simulations=15, rng=np.random.default_rng(seed))
        buf.add_game(result.examples)
    assert len(buf) > 0
    obs, policy, value = buf.sample_batch(8)
    assert obs.shape[1:] == (2, 3, 3)
    assert policy.shape[1] == 9
    assert value.ndim == 1
