"""CLI: AlphaZero self-play training (mirrors raccoon's scripts/train.py).

Examples:
    python scripts/train_alphazero.py --game tictactoe --iterations 40 \
        --games-per-iter 30 --simulations 50 --eval-every 5
    python scripts/train_alphazero.py --game dice_race --iterations 30 --simulations 40
    python scripts/train_alphazero.py --game connect4 --iterations 60 --simulations 80
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from azl.evaluate import mcts_player, play_match, random_player, solver_player
from azl.games import get_game
from azl.metrics import optimal_action_accuracy, value_mae_vs_solver
from azl.network import load_checkpoint, net_for_game, save_checkpoint
from azl.solvers import ExactSolver, enumerate_states
from azl.trainer import Coach, CoachConfig

SOLVABLE = {"tictactoe", "dice_race"}


def build_eval_callback(game_name, game_cls, eval_every, eval_sims, seed):
    if eval_every <= 0:
        return None
    rng = np.random.default_rng(seed + 999)

    if game_name in SOLVABLE:
        solver = ExactSolver()
        states = enumerate_states(game_cls.start(), include_chance=False)
        if len(states) > 600:
            idx = rng.choice(len(states), size=600, replace=False)
            states = [states[i] for i in idx]

        def cb(i, metrics, coach):
            if i % eval_every:
                return
            play = mcts_player(coach.net, num_simulations=eval_sims, rng=np.random.default_rng(i))
            acc = optimal_action_accuracy(play, states, solver)
            mae = value_mae_vs_solver(coach.net, states, solver)
            print(f"    [eval] optimal-move acc {acc:.3f}   value MAE vs solver {mae:.3f}")
    else:
        def cb(i, metrics, coach):
            if i % eval_every:
                return
            res = play_match(
                game_cls, mcts_player(coach.net, eval_sims, rng=np.random.default_rng(i)),
                random_player(rng), num_games=40, rng=rng,
            )
            print(f"    [eval] vs random: {res['a_wins']}-{res['draws']}-{res['a_losses']} "
                  f"(ret {res['a_return_mean']:+.2f})")
    return cb


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--game", default="tictactoe")
    p.add_argument("--iterations", type=int, default=40)
    p.add_argument("--games-per-iter", type=int, default=20)
    p.add_argument("--simulations", type=int, default=50)
    p.add_argument("--c-puct", type=float, default=1.5)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--training-steps", type=int, default=40)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--replay-size", type=int, default=50_000)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--temp-threshold", type=int, default=4)
    p.add_argument("--dirichlet-alpha", type=float, default=0.0)
    p.add_argument("--noise-eps", type=float, default=0.25)
    p.add_argument("--value-bootstrap-alpha", type=float, default=1.0)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--channels", type=int, default=32)
    p.add_argument("--num-blocks", type=int, default=3)
    p.add_argument("--trunk", default=None, choices=[None, "mlp", "resnet"])
    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--eval-sims", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default="runs")
    p.add_argument("--resume", default=None, help="checkpoint to warm-start from (e.g. a pretrained net)")
    args = p.parse_args()

    game_cls = get_game(args.game)
    if args.resume:
        net = load_checkpoint(args.resume)
        print(f"resumed from {args.resume}")
    else:
        net_kwargs = {"hidden": args.hidden, "channels": args.channels, "num_blocks": args.num_blocks}
        net = net_for_game(game_cls, trunk=args.trunk, **net_kwargs)

    os.makedirs(args.out_dir, exist_ok=True)
    log_path = os.path.join(args.out_dir, f"{args.game}_log.jsonl")
    cfg = CoachConfig(
        games_per_iter=args.games_per_iter, num_simulations=args.simulations, c_puct=args.c_puct,
        temperature=args.temperature, temp_threshold=args.temp_threshold,
        dirichlet_alpha=args.dirichlet_alpha, noise_eps=args.noise_eps,
        value_bootstrap_alpha=args.value_bootstrap_alpha, replay_size=args.replay_size,
        batch_size=args.batch_size, training_steps=args.training_steps,
        lr=args.lr, weight_decay=args.weight_decay,
    )
    coach = Coach(net=net, game_cls=game_cls, config=cfg, seed=args.seed, log_path=log_path)

    cb = build_eval_callback(args.game, game_cls, args.eval_every, args.eval_sims, args.seed)
    coach.train(args.iterations, callback=cb)

    ckpt = os.path.join(args.out_dir, f"{args.game}_final.pt")
    save_checkpoint(net, ckpt)
    print(f"saved checkpoint -> {ckpt}\nlog -> {log_path}")


if __name__ == "__main__":
    main()
