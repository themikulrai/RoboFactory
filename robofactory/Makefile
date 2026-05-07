# Convenience targets for the openpi-robofactory project.
#
# Run from the robofactory/ directory:
#     make manifest          # filesystem-only (no network)
#     make manifest-wandb    # filesystem + wandb
#     make manifest-dry      # dry-run, no write
#     make test              # run all unit tests
#     make test-paths
#     make test-manifest
#     make test-layout

PYTHON ?= /iris/u/mikulrai/data/miniforge3/envs/RoboFactory/bin/python

.PHONY: manifest manifest-wandb manifest-dry \
        test test-paths test-manifest test-layout

manifest:
	$(PYTHON) -m robofactory.utils.manifest --source filesystem

manifest-wandb:
	$(PYTHON) -m robofactory.utils.manifest --source both

manifest-dry:
	$(PYTHON) -m robofactory.utils.manifest --source filesystem --dry-run

test: test-paths test-manifest test-layout

test-paths:
	$(PYTHON) -m unittest robofactory.utils.test_paths -v

test-manifest:
	$(PYTHON) -m unittest robofactory.utils.test_manifest -v

test-layout:
	$(PYTHON) -m unittest robofactory.scripts.test_layout -v
