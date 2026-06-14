# Convenience targets. On Windows without `make`, run the underlying commands directly
# (see README "Quick start"). Mirrors the workflow of the raccoon project.

PY ?= python

setup:           ## create venv + editable install with dev deps
	$(PY) -m venv .venv
	.venv/Scripts/python -m pip install -U pip
	.venv/Scripts/python -m pip install -e .[dev]

test:            ## run the test suite
	$(PY) -m pytest

lab:             ## launch JupyterLab on the notebooks/ folder
	$(PY) -m jupyterlab notebooks

smoke:           ## tiny end-to-end AlphaZero run on tic-tac-toe (sanity check)
	$(PY) scripts/train_alphazero.py --game tictactoe --iterations 3 --games-per-iter 8 --simulations 25 --eval-every 0

train-az:        ## AlphaZero self-play training (override flags as needed)
	$(PY) scripts/train_alphazero.py --game tictactoe --iterations 40 --games-per-iter 30 --simulations 50

train-sup:       ## supervised foundations training (ResNet on MNIST)
	$(PY) scripts/train_supervised.py --model resnet --dataset mnist --epochs 3

pretrain:        ## supervised value-head pretraining from the exact solver
	$(PY) scripts/pretrain.py --game tictactoe --epochs 5

.PHONY: setup test lab smoke train-az train-sup pretrain
