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
#     make investigate RUN_ID=<id> OUT_DIR=<path>     # workflow#1 diagnostics
#     make investigate-explicit CKPT=... DATASET=... SCENE=... OUT_DIR=...

PYTHON ?= /iris/u/mikulrai/data/miniforge3/envs/RoboFactory/bin/python

.PHONY: manifest manifest-wandb manifest-dry \
        test test-paths test-manifest test-layout \
        investigate investigate-explicit

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

# Workflow improvement #1 — diagnostic battery to run BEFORE retraining.
# Looks up artifacts via manifest run_id, runs the heavy preflights, prints
# a verdict table, exits non-zero if any check failed.
investigate:
ifndef RUN_ID
	$(error RUN_ID is required; e.g. `make investigate RUN_ID=dp_ws_pm_in1k_cent_eval_c300_s100 OUT_DIR=/iris/u/mikulrai/runs/investigate/...`)
endif
ifndef OUT_DIR
	$(error OUT_DIR is required)
endif
	$(PYTHON) -m robofactory.scripts.diagnostics.investigate \
	    --run-id $(RUN_ID) --out-dir $(OUT_DIR)

# Same battery, but pass the artifacts directly when not in the manifest.
investigate-explicit:
ifndef CKPT
	$(error CKPT, DATASET, SCENE, OUT_DIR all required)
endif
ifndef DATASET
	$(error DATASET is required)
endif
ifndef SCENE
	$(error SCENE is required)
endif
ifndef OUT_DIR
	$(error OUT_DIR is required)
endif
	$(PYTHON) -m robofactory.scripts.diagnostics.investigate \
	    --ckpt $(CKPT) --dataset $(DATASET) --scene-config $(SCENE) \
	    --out-dir $(OUT_DIR)
