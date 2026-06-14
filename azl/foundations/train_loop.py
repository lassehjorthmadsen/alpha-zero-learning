"""A small, generic supervised training loop reused across A1-A4.

``fit`` runs the epoch loop, optimiser, device handling and history tracking.
What loss to compute is pluggable via a ``compute_loss`` callback with signature

    compute_loss(model, batch, device) -> (loss, batch_size, extra)

where ``extra`` maps metric-name -> summed-value-over-batch (e.g. ``correct``).
Two ready-made callbacks are provided: :func:`classification_loss` (A1-A3) and
:func:`make_two_head_loss` (A4 -- the cross-entropy + MSE combo that mirrors
raccoon's ``policy_loss + value_loss``).
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from azl.foundations.data import digit_to_value

ComputeLoss = Callable[[nn.Module, tuple, torch.device], tuple[torch.Tensor, int, dict]]


def default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def classification_loss(model: nn.Module, batch, device) -> tuple[torch.Tensor, int, dict]:
    """Cross-entropy + top-1 accuracy for a single-output classifier."""
    x, y = batch
    x, y = x.to(device), y.to(device)
    logits = model(x)
    loss = F.cross_entropy(logits, y)
    correct = (logits.argmax(1) == y).sum().item()
    return loss, y.size(0), {"correct": correct}


def make_two_head_loss(value_weight: float = 1.0) -> ComputeLoss:
    """Build a loss for :class:`TwoHeadedNet`: CE(class) + value_weight * MSE(value).

    This is the toy twin of raccoon's training objective. The digit label is the
    classification target; ``digit_to_value(label)`` is the regression target.
    """

    def compute(model: nn.Module, batch, device) -> tuple[torch.Tensor, int, dict]:
        x, y = batch
        x, y = x.to(device), y.to(device)
        value_target = digit_to_value(y).to(device)
        class_logits, value = model(x)
        class_loss = F.cross_entropy(class_logits, y)          # like policy loss
        value_loss = F.mse_loss(value, value_target)           # like value loss
        loss = class_loss + value_weight * value_loss
        correct = (class_logits.argmax(1) == y).sum().item()
        return loss, y.size(0), {
            "correct": correct,
            "class_loss": class_loss.item() * y.size(0),
            "value_loss": value_loss.item() * y.size(0),
        }

    return compute


@torch.no_grad()
def evaluate(model: nn.Module, loader, compute_loss: ComputeLoss = classification_loss, device=None) -> dict:
    """Average loss + metrics over a loader (no gradient)."""
    device = device or default_device()
    model.eval()
    total_loss, n = 0.0, 0
    sums: dict[str, float] = {}
    for batch in loader:
        loss, bs, extra = compute_loss(model, batch, device)
        total_loss += loss.item() * bs
        n += bs
        for k, v in extra.items():
            sums[k] = sums.get(k, 0.0) + v
    out = {"loss": total_loss / max(n, 1)}
    for k, v in sums.items():
        out[k.replace("correct", "acc")] = v / max(n, 1)
    return out


def fit(
    model: nn.Module,
    train_loader,
    val_loader=None,
    *,
    epochs: int = 3,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device=None,
    compute_loss: ComputeLoss = classification_loss,
    optimizer=None,
    verbose: bool = True,
) -> dict:
    """Train ``model`` and return a history dict of per-epoch metrics."""
    device = device or default_device()
    model.to(device)
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    history: dict[str, list] = {"train_loss": [], "val_loss": [], "val_acc": []}
    for epoch in range(1, epochs + 1):
        model.train()
        running, n = 0.0, 0
        for batch in train_loader:
            loss, bs, _ = compute_loss(model, batch, device)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += loss.item() * bs
            n += bs
        train_loss = running / max(n, 1)
        history["train_loss"].append(train_loss)

        msg = f"epoch {epoch:>2}/{epochs}  train_loss {train_loss:.4f}"
        if val_loader is not None:
            val = evaluate(model, val_loader, compute_loss, device)
            history["val_loss"].append(val["loss"])
            history["val_acc"].append(val.get("acc"))
            msg += f"  val_loss {val['loss']:.4f}"
            if val.get("acc") is not None:
                msg += f"  val_acc {val['acc']:.4f}"
        if verbose:
            print(msg)
    return history
