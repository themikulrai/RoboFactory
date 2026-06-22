"""Subtask vocabulary for subtask-conditioned data generation.

IMPORT-CLEAN: this module is *pure python* and MUST NOT import sapien / numpy /
robofactory.planner at module top, so its unit tests run off-GPU (login node).
The only filesystem touch is a LAZY yaml read inside ``assert_cube_color_map``.

Vocabulary design (see plan ``1-this-is-not-witty-quokka.md``):

* VERBS: 6 verbs, ids FIXED. ``wait`` keeps the arm idle (frozen qpos); the rest
  are the primitive verbs. ``place``=5 was *appended* to the pre-existing LB
  vocab (0..4) so existing LB configs/tests keep working.
* TARGET = a parameter, not a verb. Each task has its own target enum. ``none``
  is always id 0 (used by ``wait`` / gripper verbs that have no spatial target).
* PROMPT = (verb, target) rendered to a short natural-language string, recorded
  per-frame per-arm alongside the small ints ``verb_id`` / ``target_id``.

The (verb_id, target_id, text) triple is what the SubtaskRecorder writes per tick.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

# --------------------------------------------------------------------------- verbs
# IDs are FIXED and load-bearing. Do NOT renumber 0..4 (existing LB configs/tests
# depend on them). ``place`` was appended as 5.
WAIT = 0
APPROACH = 1
OPEN_GRIPPER = 2
CLOSE_GRIPPER = 3
LIFT = 4
PLACE = 5

# Ordered (id == index) list of canonical verb names.
VERBS: List[str] = [
    "wait",          # 0
    "approach",      # 1
    "open_gripper",  # 2
    "close_gripper", # 3
    "lift",          # 4
    "place",         # 5
]

# Reverse map name -> id (for callers that build programs by name).
VERB_IDS: Dict[str, int] = {name: i for i, name in enumerate(VERBS)}

# --------------------------------------------------------------------------- targets
# Per-task target enums. id 0 is always "none" (no spatial target).
#
# LiftBarrier: the barrier has 4 annotated contact points (ids 0..3); the solver
# grasps with contact ids 1 and 2 (see solutions/lift_barrier.py). We expose those
# as ``left_end`` / ``right_end``. The *contact-point id* used by the planner is
# carried separately (LB_TARGET_CONTACT_ID) so target_id stays a small dense enum.
LB_TARGETS: Dict[str, int] = {
    "none": 0,
    "left_end": 1,
    "right_end": 2,
}
# target_id -> barrier annotation contact_points_pose id used by
# get_grasp_pose_w_labeled_direction(..., id=<this>).
LB_TARGET_CONTACT_ID: Dict[int, int] = {
    LB_TARGETS["left_end"]: 1,
    LB_TARGETS["right_end"]: 2,
}

# ThreeRobotsStackCube: cubes by COLOR + stack destinations.
TSC_TARGETS: Dict[str, int] = {
    "none": 0,
    "blue": 1,
    "green": 2,
    "red": 3,
    "goal_region": 4,
    "on_blue": 5,
    "on_green": 6,
    "on_red": 7,
}

# Static cube->color map. NOT exposed in the env -> hardcoded here and ASSERTED
# against the yaml RGB arrays by assert_cube_color_map (fail loudly on drift).
TSC_CUBE_COLOR: Dict[str, str] = {
    "cubeA": "blue",
    "cubeB": "green",
    "cubeC": "red",
}
# Inverse: color -> cube actor name (the "green cube" == cubeB wherever it is).
TSC_COLOR_CUBE: Dict[str, str] = {v: k for k, v in TSC_CUBE_COLOR.items()}

# Per-task target enum lookup.
TASK_TARGETS: Dict[str, Dict[str, int]] = {
    "LiftBarrier": LB_TARGETS,
    "ThreeRobotsStackCube": TSC_TARGETS,
}


def target_name(task: str, target_id: int) -> str:
    """Reverse-lookup the target *name* for a (task, target_id)."""
    targets = TASK_TARGETS[task]
    for name, tid in targets.items():
        if tid == target_id:
            return name
    raise KeyError(f"unknown target_id {target_id} for task {task!r}")


def target_id(task: str, name: str) -> int:
    """Look up the target id for a (task, target name)."""
    return TASK_TARGETS[task][name]


# --------------------------------------------------------------------------- prompts
# Rendered natural-language templates. Keyed by (task, verb_id). ``{target}`` is
# filled with a per-task human phrase for the target (see _target_phrase). Verbs
# with no spatial target (wait / gripper) ignore {target}.
PROMPT_TEMPLATES: Dict[str, Dict[int, str]] = {
    "LiftBarrier": {
        WAIT: "wait",
        APPROACH: "grasp the {target}",
        OPEN_GRIPPER: "open the gripper",
        CLOSE_GRIPPER: "close the gripper",
        LIFT: "lift the {target}",
        PLACE: "place the {target}",
    },
    "ThreeRobotsStackCube": {
        WAIT: "wait",
        APPROACH: "approach the {target}",
        OPEN_GRIPPER: "open the gripper",
        CLOSE_GRIPPER: "close the gripper",
        LIFT: "lift the {target}",
        PLACE: "place {target}",
    },
}

# Human phrase per (task, target_id). For LIFT/PLACE the rendered noun reuses the
# approach noun where sensible; PLACE destinations get their own phrasing.
_LB_TARGET_PHRASE: Dict[int, str] = {
    LB_TARGETS["none"]: "barrier",
    LB_TARGETS["left_end"]: "left end",
    LB_TARGETS["right_end"]: "right end",
}
_TSC_TARGET_PHRASE: Dict[int, str] = {
    TSC_TARGETS["none"]: "cube",
    TSC_TARGETS["blue"]: "blue cube",
    TSC_TARGETS["green"]: "green cube",
    TSC_TARGETS["red"]: "red cube",
    TSC_TARGETS["goal_region"]: "on the goal",
    TSC_TARGETS["on_blue"]: "on the blue cube",
    TSC_TARGETS["on_green"]: "on the green cube",
    TSC_TARGETS["on_red"]: "on the red cube",
}
_TASK_TARGET_PHRASE: Dict[str, Dict[int, str]] = {
    "LiftBarrier": _LB_TARGET_PHRASE,
    "ThreeRobotsStackCube": _TSC_TARGET_PHRASE,
}


def _target_phrase(task: str, target_id_: int) -> str:
    phrases = _TASK_TARGET_PHRASE[task]
    if target_id_ not in phrases:
        raise KeyError(f"no target phrase for target_id {target_id_} in task {task!r}")
    return phrases[target_id_]


def render(verb_id: int, target_id_: int, task: str) -> str:
    """Render a (verb_id, target_id) pair to its NL prompt for ``task``.

    Examples (ThreeRobotsStackCube):
        render(APPROACH, TSC_TARGETS['green'], 'ThreeRobotsStackCube')
            -> "approach the green cube"
        render(PLACE, TSC_TARGETS['goal_region'], 'ThreeRobotsStackCube')
            -> "place on the goal"
        render(WAIT, 0, 'ThreeRobotsStackCube') -> "wait"
    Examples (LiftBarrier):
        render(APPROACH, LB_TARGETS['right_end'], 'LiftBarrier')
            -> "grasp the right end"
    """
    if task not in PROMPT_TEMPLATES:
        raise KeyError(f"unknown task {task!r}; known: {sorted(PROMPT_TEMPLATES)}")
    templates = PROMPT_TEMPLATES[task]
    if verb_id not in templates:
        raise KeyError(f"unknown verb_id {verb_id} for task {task!r}")
    template = templates[verb_id]
    if "{target}" in template:
        return template.format(target=_target_phrase(task, target_id_))
    return template


# --------------------------------------------------------------------------- yaml assert

# Reference RGB (0..1) for each cube color name. Matched against the yaml with a
# tolerance so float formatting differences don't trip the guard.
_COLOR_RGB_REF: Dict[str, Tuple[float, float, float]] = {
    "blue": (0.04705882, 0.16470588, 0.62745098),
    "green": (0.0, 1.0, 0.0),
    "red": (1.0, 0.0, 0.0),
}


def _dominant_channel(rgb) -> int:
    """Index (0=R,1=G,2=B) of the largest of the first three channels."""
    r, g, b = float(rgb[0]), float(rgb[1]), float(rgb[2])
    return max(range(3), key=lambda i: (r, g, b)[i])


def assert_cube_color_map(
    yaml_path: str | None = None, atol: float = 0.02
) -> Dict[str, str]:
    """Assert TSC_CUBE_COLOR matches the RGB arrays in three_robots_stack_cube.yaml.

    LAZY import of yaml so the module stays import-clean. Raises AssertionError
    (fail loudly) on ANY drift between the static map and the yaml colors.

    Returns the validated {cube_name: color_name} map on success.

    The check is two-pronged so it catches both kinds of drift:
      1. the DOMINANT RGB channel of each cube must match the color name
         (blue->B, green->G, red->R), and
      2. the full RGB triple must be within ``atol`` of the reference.
    """
    import os
    import yaml  # lazy: keep module import-clean (no yaml at top)

    if yaml_path is None:
        # configs/table/three_robots_stack_cube.yaml relative to this file:
        # robofactory/planner/ -> robofactory/configs/table/...
        here = os.path.dirname(os.path.abspath(__file__))
        yaml_path = os.path.join(
            here, os.pardir, "configs", "table", "three_robots_stack_cube.yaml"
        )
        yaml_path = os.path.normpath(yaml_path)

    with open(yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.load(f.read(), Loader=yaml.FullLoader)

    # Collect cube -> rgb from the scene primitives.
    primitives = cfg.get("scene", {}).get("primitives", [])
    cube_rgb: Dict[str, list] = {}
    for prim in primitives:
        name = prim.get("name")
        if name in TSC_CUBE_COLOR:
            color = prim.get("params", {}).get("color")
            if color is None:
                raise AssertionError(
                    f"cube {name!r} in {yaml_path} has no params.color"
                )
            cube_rgb[name] = color

    missing = set(TSC_CUBE_COLOR) - set(cube_rgb)
    if missing:
        raise AssertionError(
            f"cubes {sorted(missing)} not found in {yaml_path} primitives; "
            f"found {sorted(cube_rgb)}"
        )

    _chan_name = {0: "red", 1: "green", 2: "blue"}
    for cube, expected_color in TSC_CUBE_COLOR.items():
        rgb = cube_rgb[cube]
        # 1) dominant-channel agreement
        dom = _chan_name[_dominant_channel(rgb)]
        if dom != expected_color:
            raise AssertionError(
                f"COLOR-MAP DRIFT: {cube} yaml color={rgb} has dominant channel "
                f"'{dom}' but TSC_CUBE_COLOR says {expected_color!r}. Update "
                f"TSC_CUBE_COLOR or the yaml."
            )
        # 2) full-triple closeness to the reference
        ref = _COLOR_RGB_REF[expected_color]
        diffs = [abs(float(rgb[i]) - ref[i]) for i in range(3)]
        if max(diffs) > atol:
            raise AssertionError(
                f"COLOR-MAP DRIFT: {cube} yaml color={rgb[:3]} not within atol="
                f"{atol} of reference {ref} for {expected_color!r} "
                f"(max channel diff {max(diffs):.4f})."
            )

    return dict(TSC_CUBE_COLOR)


# --------------------------------------------------------------------------- inline unit tests
# Run with:  python -m robofactory.planner.subtask_vocab   (pure python, no sapien)
if __name__ == "__main__":
    # verb ids fixed
    assert VERBS[0] == "wait" and VERBS[5] == "place", VERBS
    assert VERB_IDS == {
        "wait": 0, "approach": 1, "open_gripper": 2,
        "close_gripper": 3, "lift": 4, "place": 5,
    }, VERB_IDS

    # render: every LB combo
    assert render(WAIT, 0, "LiftBarrier") == "wait"
    assert render(APPROACH, LB_TARGETS["left_end"], "LiftBarrier") == "grasp the left end"
    assert render(APPROACH, LB_TARGETS["right_end"], "LiftBarrier") == "grasp the right end"
    assert render(LIFT, LB_TARGETS["left_end"], "LiftBarrier") == "lift the left end"
    assert render(CLOSE_GRIPPER, 0, "LiftBarrier") == "close the gripper"
    assert render(OPEN_GRIPPER, 0, "LiftBarrier") == "open the gripper"

    # render: TSC combos
    assert render(APPROACH, TSC_TARGETS["green"], "ThreeRobotsStackCube") == "approach the green cube"
    assert render(APPROACH, TSC_TARGETS["red"], "ThreeRobotsStackCube") == "approach the red cube"
    assert render(APPROACH, TSC_TARGETS["blue"], "ThreeRobotsStackCube") == "approach the blue cube"
    assert render(PLACE, TSC_TARGETS["goal_region"], "ThreeRobotsStackCube") == "place on the goal"
    assert render(PLACE, TSC_TARGETS["on_blue"], "ThreeRobotsStackCube") == "place on the blue cube"
    assert render(WAIT, 0, "ThreeRobotsStackCube") == "wait"

    # target round-trips
    assert target_name("LiftBarrier", 2) == "right_end"
    assert target_id("ThreeRobotsStackCube", "green") == 2
    assert LB_TARGET_CONTACT_ID[LB_TARGETS["left_end"]] == 1
    assert LB_TARGET_CONTACT_ID[LB_TARGETS["right_end"]] == 2
    assert TSC_COLOR_CUBE["green"] == "cubeB"

    # yaml color assert (reads the real yaml lazily)
    out = assert_cube_color_map()
    assert out == {"cubeA": "blue", "cubeB": "green", "cubeC": "red"}, out

    print("subtask_vocab inline tests OK")
