# CLAUDE.md

Guidance for Claude Code when working in this repository.

## What this is

`alpha-zero-learning` (package `azl`) is a **from-scratch teaching companion** to
**raccoon**, the user's AlphaZero-style backgammon AI. It exists so the user can
build first-principles intuition for AlphaZero training on tiny, fast, *solvable*
toy problems before guiding raccoon's development. Optimise for **clarity and
tweakability over performance** — every line should be readable, and runs should
finish in seconds-to-minutes on CPU.

raccoon itself lives on WSL at `//wsl.localhost/Ubuntu/home/lasse/raccoon/`
(docs: https://lassehjorthmadsen.github.io/raccoon/). When changing things here,
keep the mapping to raccoon faithful — see `notebooks/B7_map_to_raccoon.ipynb`.

## Environment & commands

Native Windows, Python 3.11, virtualenv in `.venv`. There is no guarantee `make`
exists on Windows, so call the venv Python directly:

```bash
.venv/Scripts/python -m pytest                  # run the test suite (should be all green)
.venv/Scripts/python -m jupyterlab notebooks    # open the curriculum
.venv/Scripts/python scripts/train_alphazero.py --game tictactoe --iterations 40 --eval-every 5
.venv/Scripts/python scripts/train_supervised.py --model resnet --epochs 3
.venv/Scripts/python scripts/pretrain.py --game dice_race --heads both --epochs 30
.venv/Scripts/python scripts/evaluate.py --game tictactoe --checkpoint runs/tictactoe_final.pt --opponent solver
```

(`make setup|test|lab|smoke` wrap these if `make` is available.)

Always run Python via `.venv/Scripts/python` (the package is installed editable as
`azl`). After changing the library, run `pytest` before touching notebooks.

## Layout

```
azl/                  reusable library
  foundations/        Track A: blocks.py (ResidualBlock/ConvTrunk), models.py
                      (MLP/SmallResNet/SeqRNN/TwoHeadedNet), data.py (MNIST views),
                      train_loop.py (generic fit/evaluate)
  games/              base.py (GameState protocol) + tictactoe / connect4 / dice_race
  solvers.py          ExactSolver (negamax + chance) -> ground truth
  network.py          AZNet: shared trunk (mlp|resnet) + policy & value heads
  mcts.py             PUCT MCTS, chance-node sampling, Dirichlet, temperature, select_action
  selfplay.py         play_one_game -> TrainingExample(obs, policy, value)
  replay_buffer.py    circular buffer
  trainer.py          Coach + CoachConfig (self-play -> train -> repeat)
  evaluate.py         players (random/solver/net/mcts) + play_match
  metrics.py          visit entropy + grading vs the solver
notebooks/            A1-A4 (foundations), B1-B7 (AlphaZero) — numbered, plotted
scripts/              CLI entry points (mirror raccoon/scripts)
tests/                pytest: games, solvers, network, mcts, selfplay/replay
```

## Invariants to preserve (don't break these)

- **`GameState` is a small protocol** (`azl/games/base.py`): `current_player`,
  `legal_actions`, `apply_action` (returns a **new** immutable state — no mutation),
  `is_terminal`, `returns` (zero-sum `[r0, r1]`), `encode`, `key`, plus chance hooks
  `is_chance`/`chance_outcomes`. New games implement exactly this.
- **Encoding is always from the current player's perspective** ("my" vs "their"),
  like raccoon's encoder.
- **Value targets are scaled to [-1, 1]** by dividing by `game.max_abs_return`
  (1 for win/lose games, 2 for `dice_race` gammons). The value head is `tanh`.
- **Loss = cross-entropy(policy) + MSE(value)** (+ weight decay). Keep both heads.
- **MCTS root must be a decision node.** Self-play advances real dice via
  `advance_through_chance` before calling `MCTS.search`; chance nodes are sampled and
  skipped inside the search, never evaluated by the network.
- **Backup sign rule:** value is flipped whenever a node's player differs from the
  leaf's player. This one rule handles chance nodes correctly (they don't change the
  player) — don't special-case it.
- **The solver is the ground truth.** Prefer grading against `ExactSolver`
  (optimal-move accuracy, value MAE, "never beats perfect play") over reading
  training loss, for the solvable games (tic-tac-toe, `dice_race`).

## Gotchas

- `dice_race` deliberately has **no hitting/contact** (independent races) so it stays
  exactly solvable. It teaches chance-node *mechanics*, not backgammon's positional
  richness. Don't "fix" this without re-checking the solver stays correct (hitting
  would make the state graph cyclic).
- Notebooks are executed with `nbconvert`; their working directory is `notebooks/`,
  so MNIST downloads to `notebooks/data/` (git-ignored). Track-A notebooks subsample
  MNIST for speed — drop `num_train` for full-dataset accuracy.
- `data/`, `runs/`, `*.pt`, `*.jsonl`, `*.egg-info/` are git-ignored.
- Keep notebooks runnable end-to-end and reasonably fast; each lesson should end with
  a short "Maps to raccoon" note.
