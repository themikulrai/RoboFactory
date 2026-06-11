"""PR9: trajectory saving for future self-training (--save-trajectory).

Shared helper that wraps an eval env in RecordEpisodeMA with the storage
discipline the audit asks for (solution_E6.md §PR9):

  * record env_states + actions + proprio (obs/agent/.../qpos), NOT RGB by
    default (RGB is re-renderable from env_states with shader_pack=default, so
    keeping it out keeps the h5 small).
  * write under a symlinked root outside the project tree
    (/iris/u/mikulrai/data/eval_trajs/, memory feedback_symlink_heavy_files) so
    heavy data never lands in git.
  * NO video from this wrapper (the drivers write their own tiled MP4s).

CAPTURE-TRAP (digest A): the pi0.5 clients decode delta->absolute
(cur_qpos + delta) BEFORE env.step, so what env.step receives — and therefore
what RecordEpisodeMA buffers as `actions/panda-{i}` — is the ABSOLUTE joint
target. No converter-side delta reconstruction is needed. The recorded h5 is
directly consumable by parse_h5_to_zarr_unified.py --state-source qpos.

The produced h5 schema (per traj_N group) matches what the converter expects:
  actions/panda-{i}   (T, 8)  float32   absolute joint targets stepped into env
  obs/agent/<uid>/qpos (T+1, 9) float32  proprio (qpos rides along)
  env_states/...                          full sim state (re-render / recompute success)
  success             (T,)  bool
  terminated/truncated (T,) bool
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple

from robofactory.utils.wrappers.record import RecordEpisodeMA

# Heavy eval trajectories live OUTSIDE the project tree (symlink discipline).
# Override with --trajectory-root or $RF_EVAL_TRAJ_ROOT.
DEFAULT_TRAJECTORY_ROOT = "/iris/u/mikulrai/data/eval_trajs"


def default_trajectory_root() -> str:
    """Resolve the default eval-trajectory root (env var override > constant)."""
    return os.environ.get("RF_EVAL_TRAJ_ROOT", DEFAULT_TRAJECTORY_ROOT)


def trajectory_output_dir(root: Optional[str], label: str) -> str:
    """Per-run subdir under the trajectory root, e.g. <root>/<label>/."""
    base = root or default_trajectory_root()
    out = Path(base) / label
    out.mkdir(parents=True, exist_ok=True)
    return str(out)


def wrap_record_trajectory(
    env,
    output_dir: str,
    trajectory_name: str = "trajectory",
) -> Tuple[object, str]:
    """Wrap `env` in RecordEpisodeMA configured for self-training data capture.

    Returns (wrapped_env, h5_path). One h5 accumulates all episodes flushed on
    reset; the JSON sidecar (same stem) carries per-episode metadata incl.
    success and the reset seed.

    Storage discipline:
      save_trajectory=True       -> the eval call sites otherwise pass False
      save_video=False           -> drivers own their tiled MP4s
      record_env_state=True      -> re-render / recompute success offline
      record_observation=True    -> needed so obs/agent/.../qpos is captured
      record_rgb=False           -> drop sensor_data/*/rgb (re-renderable, big)
      record_reward=False         -> rewards unused for self-training, save space
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    wrapped = RecordEpisodeMA(
        env,
        output_dir,
        save_trajectory=True,
        trajectory_name=trajectory_name,
        save_video=False,
        record_env_state=True,
        record_observation=True,
        record_rgb=False,  # PR9: RGB re-renderable from env_states; keep h5 small
        record_reward=False,
        max_steps_per_video=30000,
        clean_on_close=False,  # keep the h5 on disk after env.close()
    )
    h5_path = str(Path(output_dir) / f"{trajectory_name}.h5")
    return wrapped, h5_path
