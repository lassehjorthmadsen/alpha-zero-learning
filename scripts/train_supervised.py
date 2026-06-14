"""CLI: Track-A supervised training on MNIST / Fashion-MNIST.

Examples:
    python scripts/train_supervised.py --model resnet --epochs 3
    python scripts/train_supervised.py --model rnn --cell lstm --epochs 3
    python scripts/train_supervised.py --model twohead --epochs 3
"""

from __future__ import annotations

import argparse

from azl.foundations.data import mnist_loaders
from azl.foundations.models import MLP, SeqRNN, SmallResNet, TwoHeadedNet
from azl.foundations.train_loop import (
    classification_loss,
    count_parameters,
    fit,
    make_two_head_loss,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="resnet", choices=["mlp", "resnet", "rnn", "twohead"])
    p.add_argument("--dataset", default="mnist", choices=["mnist", "fashion"])
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--num-train", type=int, default=None, help="subsample for a fast run")
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--channels", type=int, default=32)
    p.add_argument("--num-blocks", type=int, default=3)
    p.add_argument("--cell", default="lstm", choices=["rnn", "lstm", "gru"])
    args = p.parse_args()

    fashion = args.dataset == "fashion"
    view = "sequence" if args.model == "rnn" else "image"
    train_loader, test_loader = mnist_loaders(
        batch_size=args.batch_size, view=view, fashion=fashion, num_train=args.num_train
    )

    if args.model == "mlp":
        model, loss = MLP(hidden=args.hidden), classification_loss
    elif args.model == "resnet":
        model, loss = SmallResNet(channels=args.channels, num_blocks=args.num_blocks), classification_loss
    elif args.model == "rnn":
        model, loss = SeqRNN(hidden_size=args.hidden, cell=args.cell), classification_loss
    else:  # twohead
        model = TwoHeadedNet(channels=args.channels, num_blocks=args.num_blocks)
        loss = make_two_head_loss(value_weight=1.0)

    print(f"{args.model}: {count_parameters(model):,} parameters")
    fit(model, train_loader, test_loader, epochs=args.epochs, lr=args.lr, compute_loss=loss)


if __name__ == "__main__":
    main()
