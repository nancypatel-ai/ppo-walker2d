.PHONY: train eval test lint docker sweep demo

PYTHON ?= python
CONFIG ?= configs/base.yaml
SEED ?= 42
CHECKPOINT ?= checkpoints/base__seed42.pt

train:
	$(PYTHON) scripts/train.py --config $(CONFIG) --seed $(SEED)

eval:
	$(PYTHON) scripts/evaluate.py --config $(CONFIG) --checkpoint $(CHECKPOINT)

test:
	$(PYTHON) -m pytest

lint:
	ruff check .
	mypy src scripts

docker:
	docker build -t ppo-walker2d .

sweep:
	$(PYTHON) scripts/sweep.py --config $(CONFIG)

demo:
	$(PYTHON) scripts/app.py

plot:
	$(PYTHON) scripts/plot.py
