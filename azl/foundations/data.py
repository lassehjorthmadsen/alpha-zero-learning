"""Dataset loaders for Track A -- MNIST / Fashion-MNIST in three "views".

The same digits are served three ways so you can feed them to different
architectures without changing the data:

* ``view="image"``    -> x shape (1, 28, 28)   for the CNN / ResNet (A2)
* ``view="sequence"`` -> x shape (28, 28)      for the RNN: 28 rows as 28
                          timesteps of 28 features each (A3)
* ``view="flat"``     -> x shape (784,)        for the plain MLP (A1)

Data is downloaded to ``root`` (``data/`` by default, which is git-ignored).
``num_train`` lets you subsample for fast, tweakable notebook runs.
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# Standard MNIST normalisation constants.
_MNIST_MEAN, _MNIST_STD = 0.1307, 0.3081


def _view_transform(view: str):
    """Return a transform that maps a (1,28,28) tensor to the requested view."""
    if view == "image":
        return transforms.Lambda(lambda t: t)                 # (1, 28, 28)
    if view == "sequence":
        return transforms.Lambda(lambda t: t.squeeze(0))      # (28, 28) = (T, features)
    if view == "flat":
        return transforms.Lambda(lambda t: t.reshape(-1))     # (784,)
    raise ValueError(f"unknown view {view!r}; use image|sequence|flat")


def get_dataset(root: str = "data", train: bool = True, fashion: bool = False, view: str = "image"):
    """Return a torchvision MNIST/Fashion-MNIST dataset in the requested view."""
    tfm = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((_MNIST_MEAN,), (_MNIST_STD,)),
            _view_transform(view),
        ]
    )
    cls = datasets.FashionMNIST if fashion else datasets.MNIST
    return cls(root=root, train=train, download=True, transform=tfm)


def mnist_loaders(
    batch_size: int = 128,
    view: str = "image",
    fashion: bool = False,
    root: str = "data",
    num_train: int | None = None,
    num_test: int | None = None,
):
    """Build (train_loader, test_loader). Subsample with ``num_train``/``num_test``."""
    train_ds = get_dataset(root, train=True, fashion=fashion, view=view)
    test_ds = get_dataset(root, train=False, fashion=fashion, view=view)
    if num_train is not None:
        train_ds = Subset(train_ds, range(min(num_train, len(train_ds))))
    if num_test is not None:
        test_ds = Subset(test_ds, range(min(num_test, len(test_ds))))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def digit_to_value(labels: torch.Tensor) -> torch.Tensor:
    """Map a digit label 0..9 to a regression target in [-1, 1].

    Used by the two-headed net (A4) as a stand-in for raccoon's *value* target:
    a tanh-bounded scalar the regression head learns alongside the classifier.
    """
    return (labels.float() - 4.5) / 4.5
