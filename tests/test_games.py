"""Game-logic tests: legality, terminal detection, returns, chance handling."""

import numpy as np

from azl.games.tictactoe import TicTacToe
from azl.games.connect4 import Connect4
from azl.games.dice_race import DiceRace, N_CHECKERS, DIE_FACES, START_PIP


# --- tic-tac-toe ----------------------------------------------------------
def test_ttt_start():
    s = TicTacToe.start()
    assert s.current_player() == 0
    assert sorted(s.legal_actions()) == list(range(9))
    assert not s.is_terminal()
    assert s.encode().shape == (2, 3, 3)


def test_ttt_apply_is_immutable():
    s = TicTacToe.start()
    s2 = s.apply_action(0)
    assert s.board == (0,) * 9          # original untouched
    assert s2.board[0] == 1 and s2.current_player() == 1


def test_ttt_top_row_win_for_player0():
    s = TicTacToe.start()
    for a in [0, 3, 1, 4, 2]:           # X:0,1,2  O:3,4
        s = s.apply_action(a)
    assert s.is_terminal()
    assert s.returns() == [1.0, -1.0]


# --- connect four ---------------------------------------------------------
def test_c4_drop_lands_at_bottom():
    s = Connect4.start().apply_action(3)
    # bottom row (r=5), column 3 should hold player 0's disc (marker 1)
    assert s.board[5 * 7 + 3] == 1
    assert s.encode().shape == (2, 6, 7)


def test_c4_vertical_win():
    s = Connect4.start()
    for a in [0, 1, 0, 1, 0, 1, 0]:     # P0 stacks column 0 four high
        s = s.apply_action(a)
    assert s.is_terminal()
    assert s.returns() == [1.0, -1.0]


def test_c4_full_column_is_illegal():
    s = Connect4.start()
    for a in [0, 0, 0, 0, 0, 0]:        # fill column 0 (6 discs)
        s = s.apply_action(a)
    assert 0 not in s.legal_actions()


# --- dice race (chance nodes) --------------------------------------------
def test_dicerace_starts_at_chance_node():
    s = DiceRace.start()
    assert s.is_chance()
    outcomes = s.chance_outcomes()
    assert len(outcomes) == DIE_FACES
    assert abs(sum(p for _, p in outcomes) - 1.0) < 1e-9
    assert s.legal_actions() == []      # no decisions until the die is rolled


def test_dicerace_decision_after_roll():
    s = DiceRace.start().apply_action(2)   # roll a 2
    assert not s.is_chance()
    assert s.legal_actions() == [START_PIP]   # both checkers sit on START_PIP
    s2 = s.apply_action(START_PIP)            # move one checker 6 -> 4
    assert s2.current_player() == 1 and s2.is_chance()
    assert sorted(s2.pips0) == [START_PIP - 2, START_PIP]


def test_dicerace_bear_off_and_gammon():
    # Player 0 has one checker left on pip 1; player 1 has borne off nothing.
    s = DiceRace(pips0=(1,), pips1=(START_PIP,) * N_CHECKERS, player=0, die=3)
    s = s.apply_action(1)               # 1 <= 3 -> bear off the last checker
    assert s.is_terminal()
    assert s.returns() == [2.0, -2.0]   # gammon: loser borne off zero
    assert s.encode().shape == (2 * START_PIP + 3,)
