"""azl — alpha-zero-learning.

A small, transparent library of toy examples for building intuition about:

* the deep-learning *foundations* behind the raccoon backgammon AI
  (ResNet, RNN, multi-output policy/value nets) -- in ``azl.foundations``; and
* the *AlphaZero* machinery itself (PUCT MCTS, self-play, the training loop,
  chance-node handling for dice games) -- in the top-level modules here.

Everything is written from scratch in pure Python + NumPy + PyTorch so you can
read, run and tweak every line. See the notebooks/ folder for the guided
curriculum and ``notebooks/B7_map_to_raccoon.ipynb`` for the explicit mapping
back to the real raccoon code.
"""

__version__ = "0.1.0"
