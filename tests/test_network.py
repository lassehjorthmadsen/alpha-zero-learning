"""AZNet tests: shapes, legal-action masking, save/load, trunk selection."""

import os
import tempfile

import torch

from azl.games.tictactoe import TicTacToe
from azl.games.connect4 import Connect4
from azl.games.dice_race import DiceRace
from azl.network import AZNet, net_for_game, save_checkpoint, load_checkpoint


def test_forward_shapes_mlp():
    net = net_for_game(TicTacToe)
    x = torch.zeros((4, *TicTacToe.start().encode().shape))
    logits, value = net(x)
    assert logits.shape == (4, 9)
    assert value.shape == (4,)
    assert torch.all(value.abs() <= 1.0)


def test_forward_shapes_resnet():
    net = net_for_game(Connect4)
    assert net.trunk_kind == "resnet"
    x = torch.zeros((2, *Connect4.start().encode().shape))
    logits, value = net(x)
    assert logits.shape == (2, 7)
    assert value.shape == (2,)


def test_predict_masks_illegal_actions():
    s = TicTacToe.start().apply_action(0).apply_action(1)  # cells 0,1 taken
    net = net_for_game(TicTacToe)
    policy, value = net.predict(s.encode(), s.legal_actions())
    assert set(policy.keys()) == set(s.legal_actions())     # only legal moves present
    assert 0 not in policy and 1 not in policy
    assert abs(sum(policy.values()) - 1.0) < 1e-5
    assert -1.0 <= value <= 1.0


def test_default_trunk_selection():
    assert net_for_game(TicTacToe).trunk_kind == "mlp"
    assert net_for_game(DiceRace).trunk_kind == "mlp"
    assert net_for_game(Connect4).trunk_kind == "resnet"


def test_checkpoint_roundtrip():
    net = net_for_game(TicTacToe)
    s = TicTacToe.start()
    p1, v1 = net.predict(s.encode(), s.legal_actions())
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "net.pt")
        save_checkpoint(net, path)
        net2 = load_checkpoint(path)
    p2, v2 = net2.predict(s.encode(), s.legal_actions())
    assert abs(v1 - v2) < 1e-6
    assert all(abs(p1[a] - p2[a]) < 1e-6 for a in p1)
