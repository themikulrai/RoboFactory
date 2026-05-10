PYTHON := /iris/u/mikulrai/data/miniforge3/envs/RoboFactory/bin/python

# Test path roots picked up by every test target. Keep in one variable so
# adding a new test dir doesn't drift between fast / slow / all.
TEST_PATHS := robofactory/utils/ robofactory/policy/_shared/

.PHONY: manifest test test-fast test-slow test-all test-utils test-eval-context \
        ckpt-index lint lint-eval help

help:
	@echo "Targets:"
	@echo "  make manifest          Regenerate /iris/u/mikulrai/runs/manifest.csv"
	@echo "                         from wandb + ckpt index + runs/ directory tree."
	@echo "  make ckpt-index        Rescan /iris/u/mikulrai/checkpoints/* into ckpt_index.jsonl."
	@echo "  make test              Alias for test-fast (default daily-dev path)."
	@echo "  make test-fast         Skip @pytest.mark.slow tests (subprocess CLI / SAPIEN)."
	@echo "  make test-slow         Run only @pytest.mark.slow tests."
	@echo "  make test-all          Run the full suite including slow tests."
	@echo "  make test-utils        Run all robofactory/utils/ tests (legacy alias)."
	@echo "  make test-eval-context Run all eval-context unit + lint tests (legacy alias)."
	@echo "  make lint-eval         AST-check every eval_*.py driver uses WandbRun/EvalRunContext."

manifest:
	$(PYTHON) -m robofactory.utils.manifest_csv

ckpt-index:
	$(PYTHON) -m robofactory.utils.ckpt_resolver

# Daily dev workflow uses test-fast (~10s end-to-end). Heavy subprocess /
# SAPIEN-touching tests are gated behind @pytest.mark.slow and run via
# `make test-slow` (typically inside SLURM, not on the login node).
test-fast:
	$(PYTHON) -m pytest $(TEST_PATHS) -m "not slow" -v

test-slow:
	$(PYTHON) -m pytest $(TEST_PATHS) -m "slow" -v

test-all:
	$(PYTHON) -m pytest $(TEST_PATHS) -v

test-utils:
	$(PYTHON) -m pytest robofactory/utils/ -v

test-eval-context:
	$(PYTHON) -m pytest robofactory/policy/_shared/ -v

lint-eval:
	$(PYTHON) robofactory/scripts/lint/check_eval_drivers.py

lint: lint-eval

# `make test` stays as the canonical entry point — fast by default so CI and
# pre-commit don't accidentally pull in the 1-2 minute slow lane. Use
# `make test-all` when you actually want the full suite.
test: test-fast
