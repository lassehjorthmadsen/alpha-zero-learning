"""CLI: supervised pretraining from the exact solver (toy of raccoon's pretrain).

Trains an AZNet on exact (value [, policy]) labels for a *solved* game, producing
a warm-start checkpoint you can then continue with self-play. With ``--heads
value`` only the value head is fit (raccoon's Stage 1: a calibrated value with a
still-random policy); ``--heads both`` also distils the optimal policy.

Examples:
    python scripts/pretrain.py --game dice_race --heads both --epochs 30
    python scripts/pretrain.py --game tictactoe --heads value --epochs 10
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn.functional as F

from azl.games import get_game
from azl.metrics import value_mae_vs_solver
from azl.network import net_for_game, save_checkpoint
from azl.solvers import ExactSolver, solver_dataset


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--game", default="dice_race")
    p.add_argument("--heads", default="both", choices=["value", "both"])
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--channels", type=int, default=32)
    p.add_argument("--num-blocks", type=int, default=3)
    p.add_argument("--trunk", default=None, choices=[None, "mlp", "resnet"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    game_cls = get_game(args.game)
    solver = ExactSolver()
    obs, values, policies, states = solver_dataset(game_cls.start(), solver)
    print(f"{args.game}: {len(states)} decision states for supervised labels")

    net = net_for_game(game_cls, trunk=args.trunk, hidden=args.hidden,
                       channels=args.channels, num_blocks=args.num_blocks)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    X = torch.from_numpy(obs).float()
    V = torch.from_numpy(values).float()
    P = torch.from_numpy(policies).float()
    n = X.size(0)

    for epoch in range(1, args.epochs + 1):
        net.train()
        perm = torch.randperm(n)
        tot = 0.0
        for start in range(0, n, args.batch_size):
            idx = perm[start:start + args.batch_size]
            logits, value = net(X[idx])
            loss = F.mse_loss(value, V[idx])
            if args.heads == "both":
                loss = loss + -(P[idx] * F.log_softmax(logits, dim=1)).sum(1).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item()
        if epoch % max(args.epochs // 10, 1) == 0 or epoch == args.epochs:
            mae = value_mae_vs_solver(net, states, solver)
            extra = ""
            if args.heads == "both":
                top1 = np.mean([
                    int(max(net.predict(s.encode(), s.legal_actions())[0],
                            key=net.predict(s.encode(), s.legal_actions())[0].get) in solver.best_actions(s))
                    for s in states
                ])
                extra = f"  policy top-1 {top1:.3f}"
            print(f"epoch {epoch:>3}  loss {tot:.3f}  value MAE {mae:.3f}{extra}")

    out = args.out or f"runs/pretrain_{args.game}.pt"
    import os
    os.makedirs(os.path.dirname(out), exist_ok=True)
    save_checkpoint(net, out)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
