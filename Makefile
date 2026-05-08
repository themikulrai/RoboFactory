PYTHON := /iris/u/mikulrai/data/miniforge3/envs/RoboFactory/bin/python

.PHONY: manifest test test-utils ckpt-index help

help:
	@echo "Targets:"
	@echo "  make manifest    Regenerate /iris/u/mikulrai/runs/manifest.csv"
	@echo "                   from wandb + ckpt index + runs/ directory tree."
	@echo "  make ckpt-index  Rescan /iris/u/mikulrai/checkpoints/* into ckpt_index.jsonl."
	@echo "  make test-utils  Run all robofactory/utils/ tests."

manifest:
	$(PYTHON) -m robofactory.utils.manifest_csv

ckpt-index:
	$(PYTHON) -m robofactory.utils.ckpt_resolver

test-utils:
	$(PYTHON) -m pytest robofactory/utils/ -v

test: test-utils
