"""Gate G2 sign-off runner: SAPIEN rendered-vs-training-frame RMSE check for Lift-Barrier.

Renders one head_camera_global frame of LiftBarrier-rf at a training episode's reset
(shader_pack="default") and compares it to that episode's frame-0 base_0_rgb_raw from the
v2.1 training dataset, via RMSE in normalized [-1, 1] space. The wrong-shader divergent
case is ~0.53; Gate G2 is GREEN when RMSE is well below ~0.1.

MUST run on a COMPUTE node (the eval hard-refuses on the login node). Example:

    srun ... /iris/u/mikulrai/data/miniforge3/envs/RoboFactory/bin/python \
        robofactory/policy/openpi_pi05/run_g2_fidelity_check.py --episode-index 0

Writes a small JSON result and prints PASS/FAIL.
"""

from __future__ import annotations

import json
from pathlib import Path

import tyro

from robofactory.policy.openpi_pi05.eval_hierarchical_lift_barrier import fidelity_check


def main(
    episode_index: int = 0,
    rmse_threshold: float = 0.1,
    out: str = "/iris/u/mikulrai/projects/RoboFactory/eval_results/g2_fidelity.json",
) -> None:
    result = fidelity_check(episode_index=episode_index, rmse_threshold=rmse_threshold)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(result, indent=2))
    print(f"[G2] wrote {out}")
    if not result["gate_g2_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    tyro.cli(main)
