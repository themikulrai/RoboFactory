"""Tests for robofactory.utils.eval_validity (solution_E6.md §PR6).

Covers the three PR6 guard groups, GPU-free:
  - VALIDITY: >5% invalid episodes -> valid:false + exit 2 + *_INVALID rename +
    wandb 'invalid' tag; the IndexError('index 15 ... size 8') port-collision
    signature maps to the wrong-policy diagnosis.
  - PROMPT: --prompt required; OOV hard-fails (prints vocab); --allow-oov-prompt
    overrides and records itself; subtask-vocab path; prompts-JSON resolution.
  - PROVENANCE: build_provenance carries every required protocol-v2 key + the
    five drivers' written JSON dicts pass a schema assertion.
"""
from __future__ import annotations

import json
import re
import warnings
from pathlib import Path

import pytest

from robofactory.utils import eval_validity as ev


REPO_ROOT = Path(__file__).resolve().parents[1]  # robofactory/
DRIVER_PATHS = {
    "eval_pi05": REPO_ROOT / "policy" / "openpi_pi05" / "eval_pi05.py",
    "eval_decent_pi05": REPO_ROOT / "policy" / "openpi_pi05" / "eval_decent_pi05.py",
    "eval_multi_dp": REPO_ROOT / "policy" / "Diffusion-Policy" / "eval_multi_dp.py",
    "eval_joint_dp": REPO_ROOT / "policy" / "Diffusion-Policy" / "eval_joint_dp.py",
    "eval_dp": REPO_ROOT / "policy" / "Diffusion-Policy" / "eval_dp.py",
}
PI05_DRIVERS = {"eval_pi05", "eval_decent_pi05"}


# --------------------------------------------------------------------------- validity
class TestClassifyValidity:
    def test_all_clean_is_valid(self):
        r = [{"seed": i, "success": 0, "steps": 100} for i in range(20)]
        out = ev.classify_validity(r)
        assert out["valid"] is True
        assert out["invalid_reason"] is None
        assert out["n_invalid"] == 0

    def test_one_failed_episode_within_threshold_valid(self):
        # 1/40 = 2.5% < 5% -> still valid.
        r = [{"seed": i, "success": 0, "steps": 100} for i in range(39)]
        r.append({"seed": 99, "success": 0, "steps": -1, "error": "boom"})
        out = ev.classify_validity(r)
        assert out["valid"] is True
        assert out["n_invalid"] == 1

    def test_over_threshold_is_invalid(self):
        # 2/20 = 10% > 5% -> invalid.
        r = [{"seed": i, "success": 0, "steps": 100} for i in range(18)]
        r += [{"seed": 18, "steps": -1, "error": "x"}, {"seed": 19, "steps": -1}]
        out = ev.classify_validity(r)
        assert out["valid"] is False
        assert "10.0%" in out["invalid_reason"]
        assert out["n_invalid"] == 2

    def test_steps_minus_one_counts_invalid(self):
        r = [{"seed": 1, "steps": -1}]  # no error key, just steps==-1
        out = ev.classify_validity(r)
        assert out["valid"] is False

    def test_nonempty_error_counts_invalid(self):
        r = [{"seed": 1, "success": 0, "steps": 50, "error": "RuntimeError(...)"}]
        out = ev.classify_validity(r)
        assert out["valid"] is False

    def test_empty_results_invalid(self):
        out = ev.classify_validity([])
        assert out["valid"] is False
        assert "no episodes" in out["invalid_reason"]

    def test_port_collision_signature_diagnosis(self):
        sig = "IndexError('index 15 is out of bounds for axis 0 with size 8')"
        r = [{"seed": 1, "steps": -1, "error": sig}]
        out = ev.classify_validity(r)
        assert out["valid"] is False
        assert "port-collision" in out["invalid_reason"]
        assert "PR2 handshake" in out["invalid_reason"]

    def test_port_collision_overrides_generic_count(self):
        # When a port-collision sig is present, that's the reason (not the count).
        sig = "IndexError('index 15 is out of bounds for axis 0 with size 8')"
        r = [{"seed": 1, "steps": -1, "error": sig}] + [
            {"seed": i, "steps": -1} for i in range(2, 5)
        ]
        out = ev.classify_validity(r)
        assert "port-collision" in out["invalid_reason"]


class TestInvalidOutputPath:
    def test_json(self):
        assert ev.invalid_output_path("/a/b/foo.json").name == "foo_INVALID.json"

    def test_jsonl(self):
        assert ev.invalid_output_path("/a/b/foo.jsonl").name == "foo_INVALID.jsonl"

    def test_idempotent(self):
        p = ev.invalid_output_path("/a/b/foo_INVALID.json")
        assert p.name == "foo_INVALID.json"

    def test_non_json_suffix(self):
        assert ev.invalid_output_path("/a/b/foo.txt").name == "foo.txt_INVALID"


class _FakeWandbRun:
    """Stub exposing .run.tags, like WandbRun."""

    def __init__(self):
        self.tags = []

    @property
    def run(self):
        return self


class TestFinalizeValidity:
    def test_valid_no_rename_no_exit(self, tmp_path: Path):
        out = tmp_path / "r.json"
        out.write_text("{}")
        validity = {"valid": True, "invalid_reason": None}
        path = ev.finalize_validity(validity, out, exit_on_invalid=True)
        assert path == out
        assert out.exists()

    def test_invalid_renames_tags_and_exits(self, tmp_path: Path):
        out = tmp_path / "r.json"
        out.write_text(json.dumps({"results": []}))
        fake = _FakeWandbRun()
        validity = {"valid": False, "invalid_reason": "boom"}
        with pytest.raises(SystemExit) as cm:
            ev.finalize_validity(validity, out, wandb_run=fake, exit_on_invalid=True)
        assert cm.value.code == ev.INVALID_EXIT_CODE  # exit 2
        # renamed
        renamed = tmp_path / "r_INVALID.json"
        assert renamed.exists() and not out.exists()
        # wandb tagged invalid
        assert ev.INVALID_TAG in fake.tags

    def test_invalid_no_exit_returns_renamed(self, tmp_path: Path):
        out = tmp_path / "r.json"
        out.write_text("{}")
        validity = {"valid": False, "invalid_reason": "boom"}
        path = ev.finalize_validity(validity, out, exit_on_invalid=False)
        assert path.name == "r_INVALID.json"
        assert path.exists()

    def test_finalize_none_wandb_safe(self, tmp_path: Path):
        out = tmp_path / "r.jsonl"
        out.write_text("x")
        validity = {"valid": False, "invalid_reason": "boom"}
        path = ev.finalize_validity(validity, out, wandb_run=None, exit_on_invalid=False)
        assert path.name == "r_INVALID.jsonl"


class TestKillServerSimulation:
    """Mirrors the kill-server-mid-eval acceptance test: once the server dies,
    every remaining episode crashes (steps==-1 + error). >5% -> invalid+exit 2."""

    def test_server_killed_midway_is_invalid(self, tmp_path: Path):
        # 10 ok, then 50 dead (server killed) -> 50/60 invalid.
        r = [{"seed": i, "success": 0, "steps": 100} for i in range(10)]
        r += [{"seed": i, "steps": -1, "error": "ConnectionClosed"} for i in range(10, 60)]
        validity = ev.classify_validity(r)
        assert validity["valid"] is False
        out = tmp_path / "eval.json"
        out.write_text(json.dumps({"validity": validity}))
        with pytest.raises(SystemExit) as cm:
            ev.finalize_validity(validity, out, exit_on_invalid=True)
        assert cm.value.code == 2
        assert (tmp_path / "eval_INVALID.json").exists()


# --------------------------------------------------------------------------- prompt
class TestPromptVocab:
    def test_required_prompt_raises(self):
        with pytest.raises(ValueError, match="REQUIRED"):
            ev.validate_prompt("", ["a", "b"], vocab_source="x")
        with pytest.raises(ValueError, match="REQUIRED"):
            ev.validate_prompt(None, ["a"], vocab_source="x")

    def test_in_vocab_ok(self):
        pv = ev.validate_prompt("a", ["a", "b"], vocab_source="x")
        assert pv["prompt_in_vocab"] is True
        assert pv["allow_oov_prompt"] is False

    def test_oov_hard_fails_and_prints_vocab(self):
        with pytest.raises(ev.PromptOutOfVocabError) as cm:
            ev.validate_prompt("zzz", ["a", "b"], vocab_source="src.json")
        msg = str(cm.value)
        assert "OUT OF VOCABULARY" in msg
        assert "Allowed prompts" in msg and "'a'" in msg

    def test_allow_oov_records_override(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pv = ev.validate_prompt("zzz", ["a"], allow_oov=True, vocab_source="x")
        assert pv["allow_oov_prompt"] is True
        assert pv["prompt_in_vocab"] is False

    def test_empty_vocab_cannot_validate_warns(self):
        with pytest.warns(UserWarning, match="no prompt vocabulary"):
            pv = ev.validate_prompt("anything", [], vocab_source="<none>")
        assert pv["prompt_in_vocab"] is None  # un-validatable, not failed

    def test_load_task_prompts_json(self):
        vocab, src = ev.load_prompt_vocab(task="three_robots_stack_cube")
        assert len(vocab) >= 1
        assert "three_robots_stack_cube_prompts.json" in src
        # the historical DEFAULT_PROMPT IS in the TSC vocab (it was a TSC string)
        assert "stack the three cubes using three robot arms" in vocab

    def test_load_from_camera_mapping_path_strips_suffix(self):
        # drivers pass --camera-mapping (a *_wristcam.json path) as `task`.
        cm = (
            "/iris/u/mikulrai/projects/openpi/examples/robofactory/"
            "camera_mappings/three_robots_stack_cube_wristcam.json"
        )
        vocab, src = ev.load_prompt_vocab(task=cm)
        assert "three_robots_stack_cube_prompts.json" in src

    def test_load_camelcase_env_id(self):
        vocab, src = ev.load_prompt_vocab(task="ThreeRobotsStackCube-rf")
        assert "three_robots_stack_cube_prompts.json" in src

    def test_load_subtask_vocab(self):
        vocab, src = ev.load_prompt_vocab(use_subtask_vocab=True)
        assert "lb_subtask_index.npz" in src
        # the 5 LB sidecar subtask labels
        for tok in ("wait", "approach", "lift"):
            assert tok in vocab

    def test_subtask_oov_for_instruction_prompt(self):
        # the LB flat instruction is deliberately OOV for the subtask vocab.
        vocab, src = ev.load_prompt_vocab(use_subtask_vocab=True)
        with pytest.raises(ev.PromptOutOfVocabError):
            ev.validate_prompt("lift the steel barrier using two robot arms", vocab, vocab_source=src)

    def test_unknown_task_returns_no_vocab(self):
        vocab, src = ev.load_prompt_vocab(task="NoSuchTask-rf")
        assert vocab == [] and src == "<none>"


# --------------------------------------------------------------------------- provenance
REQUIRED_PROV_KEYS = {
    "eval_protocol",
    "hostname",
    "gpu_name",
    "git",
    "shader_pack",
    "enable_shadow",
    "sim_backend",
    "seed_provenance",
    "prompts",
    "max_env_steps",
    "chunk_config",
}


class TestProvenance:
    def test_protocol_v2_and_required_keys(self):
        p = ev.build_provenance(
            shader_pack="default",
            enable_shadow=False,
            sim_backend="auto",
            seed_provenance={"seed_pool": "canonical_env_60", "seed_pool_sha": "abc"},
            prompts=["stack the cubes"],
            max_env_steps=400,
            chunk_config={"replan_after": 8},
            server_metadata={"port_8000": {"config_name": "x"}},
        )
        assert p["eval_protocol"] == "v2"
        assert REQUIRED_PROV_KEYS.issubset(p.keys())
        assert p["enable_shadow"] is False
        assert p["seed_provenance"]["seed_pool"] == "canonical_env_60"

    def test_both_repo_git_sha(self):
        p = ev.build_provenance(seed_provenance={}, prompts=[])
        assert set(p["git"].keys()) == {"robofactory", "openpi"}
        for repo in ("robofactory", "openpi"):
            assert "sha" in p["git"][repo] and "dirty" in p["git"][repo]

    def test_dp_records_ckpt_md5_and_zarr(self, tmp_path: Path):
        ckpt = tmp_path / "300.ckpt"
        ckpt.write_bytes(b"weights")
        p = ev.build_provenance(
            seed_provenance={}, prompts=None,
            ckpt_paths=[str(ckpt)], zarr_repo_id="d2_wristcam",
        )
        assert p["ckpt_paths"] == [str(ckpt)]
        assert p["ckpt_md5"][str(ckpt)] is not None  # 32 hex chars
        assert len(p["ckpt_md5"][str(ckpt)]) == 32
        assert p["zarr_repo_id"] == "d2_wristcam"

    def test_file_md5_missing_is_none(self):
        assert ev.file_md5("/no/such/ckpt") is None
        assert ev.file_md5(None) is None

    def test_git_provenance_unknown_repo(self):
        out = ev.git_provenance("/no/such/repo")
        assert out["sha"] == "unknown"


# ----------------------------------------------- driver wiring (source-grep, no import)
class TestDriverWiring:
    def test_default_prompt_deleted_from_pi05_drivers(self):
        # The TSC DEFAULT_PROMPT assignment must be gone (digest B B4). Allow the
        # word in comments documenting the deletion; forbid an actual assignment.
        bad = re.compile(r'^\s*DEFAULT_PROMPT\s*=\s*["\']')
        for name in PI05_DRIVERS:
            for ln in DRIVER_PATHS[name].read_text().splitlines():
                assert not bad.match(ln), f"{name}: DEFAULT_PROMPT assignment still present: {ln!r}"

    def test_pi05_drivers_require_prompt_and_validate(self):
        for name in PI05_DRIVERS:
            src = DRIVER_PATHS[name].read_text()
            assert "validate_prompt(" in src, f"{name} must validate the prompt (PR6)"
            assert "allow_oov_prompt" in src, f"{name} must expose --allow-oov-prompt"
            assert "tyro.MISSING" in src  # --prompt required

    def test_all_drivers_classify_validity_and_finalize(self):
        for name, path in DRIVER_PATHS.items():
            src = path.read_text()
            assert "classify_validity(" in src, f"{name} must classify validity (PR6)"
            assert "finalize_validity(" in src, f"{name} must finalize validity (PR6)"

    def test_all_drivers_build_provenance(self):
        for name, path in DRIVER_PATHS.items():
            assert "build_provenance(" in path.read_text(), f"{name} must build provenance (PR6)"


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
