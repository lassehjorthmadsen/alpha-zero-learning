"""CLI: evaluate a checkpoint by match play (mirrors raccoon's eval scripts).

Examples:
    python scripts/evaluate.py --game tictactoe --checkpoint runs/tictactoe_final.pt --opponent solver
    python scripts/evaluate.py --game connect4 --checkpoint runs/connect4_final.pt --opponent random
"""

from __future__ import annotations

import argparse

import numpy as np

from azl.evaluate import mcts_player, net_player, play_match, random_player, solver_player
from azl.games import get_game
from azl.network import load_checkpoint
from azl.solvers import ExactSolver


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--game", default="tictactoe")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--opponent", default="random", choices=["random", "solver", "net", "self"])
    p.add_argument("--num-games", type=int, default=100)
    p.add_argument("--sims", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    game_cls = get_game(args.game)
    net = load_checkpoint(args.checkpoint)
    rng = np.random.default_rng(args.seed)

    player_a = mcts_player(net, num_simulations=args.sims, rng=rng)
    if args.opponent == "random":
        player_b = random_player(rng)
    elif args.opponent == "solver":
        player_b = solver_player(ExactSolver(), rng)
    elif args.opponent == "net":
        player_b = net_player(net)
    else:
        player_b = mcts_player(net, num_simulations=args.sims, rng=rng)

    res = play_match(game_cls, player_a, player_b, num_games=args.num_games, rng=rng)
    print(f"A (MCTS {args.sims} sims) vs {args.opponent}: "
          f"{res['a_wins']} wins / {res['draws']} draws / {res['a_losses']} losses  "
          f"(mean return {res['a_return_mean']:+.3f} over {res['num_games']} games)")


if __name__ == "__main__":
    main()
