PYTHON := /iris/u/mikulrai/data/miniforge3/envs/RoboFactory/bin/python

.PHONY: manifest test test-utils test-eval-context ckpt-index lint lint-eval help

help:
	@echo "Targets:"
	@echo "  make manifest          Regenerate /iris/u/mikulrai/runs/manifest.csv"
	@echo "                         from wandb + ckpt index + runs/ directory tree."
	@echo "  make ckpt-index        Rescan /iris/u/mikulrai/checkpoints/* into ckpt_index.jsonl."
	@echo "  make test-utils        Run all robofactory/utils/ tests."
	@echo "  make test-eval-context Run all eval-context unit + lint tests."
	@echo "  make lint-eval         AST-check every eval_*.py driver uses WandbRun/EvalRunContext."

manifest:
	$(PYTHON) -m robofactory.utils.manifest_csv

ckpt-index:
	$(PYTHON) -m robofactory.utils.ckpt_resolver

test-utils:
	$(PYTHON) -m pytest robofactory/utils/ -v

test-eval-context:
	$(PYTHON) -m pytest robofactory/policy/_shared/ -v

lint-eval:
	$(PYTHON) robofactory/scripts/lint/check_eval_drivers.py

lint: lint-eval

test: test-utils test-eval-context
