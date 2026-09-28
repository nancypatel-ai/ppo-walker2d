.PHONY: train eval test lint docker sweep demo

PYTHON ?= python
CONFIG ?= configs/base.yaml
SEED ?= 42

train:
	$(PYTHON) scripts/train.py --config $(CONFIG) --seed $(SEED)

eval:
	$(PYTHON) scripts/evaluate.py --config $(CONFIG) --checkpoint $(checkpoint)

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
