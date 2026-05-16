"""Smoke tests for scripts/canonical/submit_matrix.sh.

All cases run with DRYRUN=1 so no SLURM submission happens. Asserts on the
emitted script body and the wrapper's stderr.
"""

from __future__ import annotations

import os
import pathlib
import subprocess

import pytest

_SCRIPT = pathlib.Path(__file__).resolve().parent / "submit_matrix.sh"
assert _SCRIPT.exists(), f"submit_matrix.sh not at {_SCRIPT}"


def _run(*args: str) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["DRYRUN"] = "1"
    return subprocess.run(
        ["bash", str(_SCRIPT), *args],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )


# ----- Argument validation


@pytest.mark.parametrize(
    "args,expected_stderr_fragment",
    [
        (["--task", "pm", "--scheme", "cent", "--cam", "ws"], "missing --policy"),
        (["--policy", "dp", "--scheme", "cent", "--cam", "ws"], "missing --task"),
        (["--policy", "dp", "--task", "pm", "--cam", "ws"], "missing --scheme"),
        (["--policy", "dp", "--task", "pm", "--scheme", "cent"], "missing --cam"),
        (
            ["--policy", "foo", "--task", "pm", "--scheme", "cent", "--cam", "ws"],
            "--policy must be",
        ),
        (
            ["--policy", "dp", "--task", "xx", "--scheme", "cent", "--cam", "ws"],
            "unknown",
        ),
        (
            ["--policy", "dp", "--task", "pm", "--scheme", "bogus", "--cam", "ws"],
            "scheme must be",
        ),
        (
            ["--policy", "dp", "--task", "pm", "--scheme", "cent", "--cam", "xy"],
            "cam must be",
        ),
        # PM is single-arm: decent is invalid
        (
            ["--policy", "dp", "--task", "pm", "--scheme", "decent", "--cam", "ws", "--arm", "0"],
            "PM is single-arm",
        ),
        # decent without --arm
        (
            ["--policy", "dp", "--task", "2sc", "--scheme", "decent", "--cam", "wc"],
            "decent requires --arm",
        ),
        # cent with --arm
        (
            ["--policy", "dp", "--task", "2sc", "--scheme", "cent", "--cam", "wc", "--arm", "0"],
            "cent does not take --arm",
        ),
        # arm out of range
        (
            ["--policy", "dp", "--task", "2sc", "--scheme", "decent", "--cam", "wc", "--arm", "5"],
            "out of range",
        ),
    ],
)
def test_arg_validation(args: list[str], expected_stderr_fragment: str) -> None:
    result = _run(*args)
    assert result.returncode != 0, f"expected nonzero exit; got {result.returncode}\nstderr: {result.stderr}"
    assert expected_stderr_fragment in result.stderr, (
        f"expected stderr to contain {expected_stderr_fragment!r}; got:\n{result.stderr}"
    )


# ----- Workspace cameras identical bug detection


def test_ws_decent_tsc_aborts_on_identical_cams() -> None:
    """TSC scene yaml has all 4 head_camera_agent* with identical poses → abort."""
    result = _run("--policy", "dp", "--task", "tsc", "--scheme", "decent", "--cam", "ws", "--arm", "0")
    assert result.returncode != 0
    assert "workspace-cameras-identical bug" in result.stderr
    assert "0/60 SR" in result.stderr


def test_ws_decent_with_override_proceeds_to_emit() -> None:
    """User can override the cam-bug abort with --allow-broken-symlink for debugging."""
    result = _run(
        "--policy", "dp", "--task", "tsc", "--scheme", "decent",
        "--cam", "ws", "--arm", "0", "--allow-broken-symlink",
    )
    assert result.returncode == 0, f"expected 0 exit with override; stderr:\n{result.stderr}"
    assert "DRYRUN=1 → not submitting" in result.stderr


def test_wc_decent_does_not_trigger_cam_bug() -> None:
    """Wristcam (per-arm own wrist) has distinct cameras → no bug."""
    result = _run(
        "--policy", "dp", "--task", "tsc", "--scheme", "decent",
        "--cam", "wc", "--arm", "0",
    )
    assert result.returncode == 0
    assert "workspace-cameras-identical bug" not in result.stderr


# ----- Hydra knob resolution


def test_pm_uses_robot_dp_yaml() -> None:
    """PM is single-arm: even cent uses robot_dp.yaml (not joint_dp)."""
    result = _run("--policy", "dp", "--task", "pm", "--scheme", "cent", "--cam", "ws")
    assert result.returncode == 0
    assert "--config-name=robot_dp.yaml" in result.stdout
    assert "joint_dp.yaml" not in result.stdout
    assert "task=default_task" in result.stdout
    assert "task.name=PickMeat-rf" in result.stdout


def test_multi_arm_cent_uses_joint_dp_yaml() -> None:
    """TSC cent should pick joint_dp.yaml + joint_task_3arm."""
    result = _run("--policy", "dp", "--task", "tsc", "--scheme", "cent", "--cam", "wc")
    assert result.returncode == 0
    assert "--config-name=joint_dp.yaml" in result.stdout
    assert "task=joint_task_3arm" in result.stdout


def test_decent_passes_current_agent_id() -> None:
    """decent arm 2 must pass current_agent_id=2 to Hydra."""
    result = _run(
        "--policy", "dp", "--task", "tsc", "--scheme", "decent",
        "--cam", "wc", "--arm", "2",
    )
    assert result.returncode == 0
    assert "current_agent_id=2" in result.stdout


def test_cam_switches_task_config() -> None:
    """--cam wc → task=default_task_wristcam; --cam ws → task=default_task."""
    wc = _run("--policy", "dp", "--task", "tsc", "--scheme", "decent", "--cam", "wc", "--arm", "0")
    assert "task=default_task_wristcam" in wc.stdout
    ws = _run("--policy", "dp", "--task", "tsc", "--scheme", "decent", "--cam", "ws", "--arm", "0", "--allow-broken-symlink")
    assert "task=default_task\n" in ws.stdout or "task=default_task " in ws.stdout


# ----- Zarr path resolution


def test_zarr_paths_match_on_disk_pattern() -> None:
    """The emitted zarr_path must match the on-disk naming convention for each task."""
    cases = [
        (["--task", "pm", "--scheme", "cent", "--cam", "ws"],
         "PickMeat-rf_workspace_cent_150.zarr"),
        (["--task", "pm", "--scheme", "cent", "--cam", "wc"],
         "PickMeat-rf_wristcam_cent_150.zarr"),
        (["--task", "2sc", "--scheme", "decent", "--cam", "ws", "--arm", "1"],
         "TwoRobotsStackCube-rf_workspace_decent_agent1_150.zarr"),
        (["--task", "tsc", "--scheme", "decent", "--cam", "wc", "--arm", "2"],
         "ThreeRobotsStackCube-rf_wristcam_decent_agent2_150.zarr"),
        # LP uses its own naming: LongPipelineDelivery_{cam}_{joint|agent<i>}.zarr
        (["--task", "lp", "--scheme", "cent", "--cam", "ws"],
         "LongPipelineDelivery_workspace_joint.zarr"),
        (["--task", "lp", "--scheme", "decent", "--cam", "wc", "--arm", "3"],
         "LongPipelineDelivery_wristcam_agent3.zarr"),
    ]
    for args, expected_zarr in cases:
        # ws-decent on multi-arm tasks needs the override flag (else cam-bug check kills it)
        if "decent" in args and "ws" in args and "pm" not in args:
            args = [*args, "--allow-broken-symlink"]
        result = _run("--policy", "dp", *args)
        assert expected_zarr in result.stdout, (
            f"args={args}\nexpected zarr {expected_zarr!r} in stdout:\n{result.stdout[:500]}"
        )


# ----- Pi0.5 path


def test_pi05_emits_canonical_config_name() -> None:
    """Pi0.5 train command must use the canonical pi05_robofactory_{task}_{cam}_{scheme} name."""
    result = _run("--policy", "pi05", "--task", "2sc", "--scheme", "decent",
                  "--cam", "wc", "--arm", "0", "--allow-broken-symlink")
    assert result.returncode == 0
    assert "pi05_robofactory_2sc_wc_decent_arm0" in result.stdout
    assert "scripts/train.py" in result.stdout


# ----- Cluster directive selection


def test_orion_cluster_sets_nice_and_account() -> None:
    """--cluster orion must produce partition=orion + account=orion + nice=10000."""
    result = _run("--policy", "dp", "--task", "pm", "--scheme", "cent", "--cam", "ws", "--cluster", "orion")
    assert result.returncode == 0
    assert "#SBATCH --partition=orion" in result.stdout
    assert "#SBATCH --account=orion" in result.stdout
    assert "#SBATCH --nice=10000" in result.stdout


def test_iris_hi_cluster_no_nice() -> None:
    """--cluster iris-hi gets account=iris and no nice directive."""
    result = _run("--policy", "dp", "--task", "pm", "--scheme", "cent", "--cam", "ws", "--cluster", "iris-hi")
    assert result.returncode == 0
    assert "#SBATCH --partition=iris-hi" in result.stdout
    assert "#SBATCH --account=iris" in result.stdout
    assert "#SBATCH --nice" not in result.stdout


# ----- Wandb name convention


def test_wandb_name_convention_decent() -> None:
    """Per dp_wandb_run_naming.md: 'Train <TASK> <CAM> DP [Decent A<i>]'."""
    result = _run("--policy", "dp", "--task", "tsc", "--scheme", "decent",
                  "--cam", "wc", "--arm", "1")
    assert result.returncode == 0
    assert 'logging.name="Train TSC WC DP Decent A1"' in result.stdout


def test_wandb_name_convention_cent() -> None:
    result = _run("--policy", "dp", "--task", "pm", "--scheme", "cent", "--cam", "ws")
    assert result.returncode == 0
    assert 'logging.name="Train PM WS DP"' in result.stdout
