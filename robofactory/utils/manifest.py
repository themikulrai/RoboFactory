"""Generate the openpi-robofactory run manifest CSV.

Plan v2 C1#9. Re-derives the manifest from the filesystem (and optionally
wandb) on every invocation. The manifest is the single grep-able index of
every train/eval run we have on disk or in wandb.

Schema (per plan v2 — 23 columns):
    run_id, model, dataset, task, encoder, scheme, arm, phase, step, seed,
    git_sha, slurm_id, status, category, wandb_url, ckpt_paths, eval_sr,
    eval_n, scene_config, camera_mapping, parent_run_id, created_utc, notes

CLI:
    python -m robofactory.utils.manifest [--output PATH] [--source SRC]
        [--dp-ckpt-root DIR] [--pi05-ckpt-root DIR]
        [--wandb-entity E] [--wandb-projects P1,P2] [--dry-run]

`--source` defaults to `filesystem` (no network). Pass `wandb` or `both` to
include wandb-side runs (requires the `wandb` package + a valid login).

Existing on-disk dir names predate the canonical run_id format from plan v2,
so for legacy rows the parser fills the structured fields it can infer from
the directory name and leaves the rest blank. Future runs that follow the
canonical naming will populate every field.
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import datetime as _dt
import re
from pathlib import Path
from typing import Iterable, Iterator, Optional

from robofactory.utils import paths

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

MANIFEST_COLUMNS: tuple[str, ...] = (
    "run_id",
    "model",
    "dataset",
    "task",
    "encoder",
    "scheme",
    "arm",
    "phase",
    "step",
    "seed",
    "git_sha",
    "slurm_id",
    "status",
    "category",
    "wandb_url",
    "ckpt_paths",
    "eval_sr",
    "eval_n",
    "scene_config",
    "camera_mapping",
    "parent_run_id",
    "created_utc",
    "notes",
)

VALID_STATUS = {"queued", "running", "done", "failed", "junked", "unknown"}
VALID_CATEGORY = {"canonical", "ablation", "junk", "unknown"}


@dataclasses.dataclass
class RunRecord:
    """One row of the manifest. All fields default empty so partial rows
    from heuristic parsers serialize cleanly."""

    run_id: str = ""
    model: str = ""
    dataset: str = ""
    task: str = ""
    encoder: str = ""
    scheme: str = ""
    arm: str = ""
    phase: str = ""
    step: str = ""
    seed: str = ""
    git_sha: str = ""
    slurm_id: str = ""
    status: str = "unknown"
    category: str = "unknown"
    wandb_url: str = ""
    ckpt_paths: str = ""
    eval_sr: str = ""
    eval_n: str = ""
    scene_config: str = ""
    camera_mapping: str = ""
    parent_run_id: str = ""
    created_utc: str = ""
    notes: str = ""

    def as_row(self) -> dict[str, str]:
        return {c: str(getattr(self, c)) for c in MANIFEST_COLUMNS}


# ---------------------------------------------------------------------------
# Canonical run_id parser (plan v2 naming convention)
# ---------------------------------------------------------------------------

_VALID_MODELS = ("dp", "pi05")
_VALID_DATASETS = ("ws", "wc")
_VALID_TASKS = ("pm", "2sc", "tsc", "lp")
_VALID_ENCODERS = (
    "in1k", "in1k-crop", "r3m", "dino-s-lora", "dino-s-spatch", "none",
)
_VALID_SCHEMES = ("cent", "decent")

_TRAIN_RE = re.compile(
    r"^(?P<model>dp|pi05)"
    r"_(?P<dataset>ws|wc)"
    r"_(?P<task>pm|2sc|tsc|lp)"
    r"_(?P<encoder>in1k-crop|in1k|r3m|dino-s-lora|dino-s-spatch|none)"
    r"_(?P<scheme>cent|decent)"
    r"(?:_arm(?P<arm>\d))?"
    r"_train"
    r"_s(?P<seed>\d+)"
    r"_(?P<git_sha>[0-9a-f]{7})$"
)

_EVAL_RE = re.compile(
    r"^(?P<model>dp|pi05)"
    r"_(?P<dataset>ws|wc)"
    r"_(?P<task>pm|2sc|tsc|lp)"
    r"_(?P<encoder>in1k-crop|in1k|r3m|dino-s-lora|dino-s-spatch|none)"
    r"_(?P<scheme>cent|decent)"
    r"(?:_arm(?P<arm>\d))?"
    r"_eval"
    r"_c(?P<step>\d+)"
    r"_s(?P<seed>\d+)"
    r"_(?P<git_sha>[0-9a-f]{7})$"
)


def parse_canonical_run_id(run_id: str) -> Optional[dict[str, str]]:
    """Parse a canonical plan-v2 run_id.

    Returns a dict of {model, dataset, task, encoder, scheme, arm, phase,
    step, seed, git_sha} on a match, else None.
    """
    for phase, pattern in (("train", _TRAIN_RE), ("eval", _EVAL_RE)):
        m = pattern.match(run_id)
        if m:
            d = m.groupdict(default="")
            d["phase"] = phase
            d.setdefault("step", "")
            d.setdefault("arm", "")
            return d
    return None


# ---------------------------------------------------------------------------
# Heuristic legacy DP dir-name parser
# ---------------------------------------------------------------------------

_DP_TASK_TOKENS = {
    "PickMeat-rf": "pm",
    "ThreeRobotsStackCube-rf": "tsc",
    "TwoRobotsStackCube-rf": "2sc",
    "LongPipeline-rf": "lp",
}


def parse_legacy_dp_dirname(name: str) -> dict[str, str]:
    """Best-effort field extraction from a legacy DP ckpt dir name.

    Examples:
        PickMeat-rf_150                                   -> task=pm
        PickMeat-rf_150_in1k_crop                         -> task=pm, encoder=in1k-crop
        ThreeRobotsStackCube-rf_agent0_d2_wristcam_150    -> task=tsc, scheme=decent, arm=0, dataset=wc
        ThreeRobotsStackCube-rf_joint_d1_workspace_150    -> task=tsc, scheme=cent, dataset=ws
        ThreeRobotsStackCube-rf_joint_d1_workspace_150_in1k -> ..., encoder=in1k
    """
    out: dict[str, str] = {"model": "dp"}
    for prefix, task in _DP_TASK_TOKENS.items():
        if name.startswith(prefix):
            out["task"] = task
            tail = name[len(prefix) + 1 :]  # drop "<prefix>_"
            break
    else:
        return out

    tokens = tail.split("_")
    arm_match = re.match(r"agent(\d)$", tokens[0]) if tokens else None
    if arm_match:
        out["scheme"] = "decent"
        out["arm"] = arm_match.group(1)
    elif tokens and tokens[0] == "joint":
        out["scheme"] = "cent"
    # dataset
    for i, tok in enumerate(tokens):
        if tok in ("d1", "d2") and i + 1 < len(tokens):
            nxt = tokens[i + 1]
            if nxt == "wristcam":
                out["dataset"] = "wc"
            elif nxt == "workspace":
                out["dataset"] = "ws"
            break
    # encoder
    joined = "_".join(tokens)
    for enc_token, canonical in (
        ("in1k_crop", "in1k-crop"),
        ("dinov2_blora", "dino-s-lora"),
        ("dinov2_spatch", "dino-s-spatch"),
        ("freezeenc", "in1k"),  # freezeenc was in1k frozen
        ("r3m", "r3m"),
        ("in1k", "in1k"),
    ):
        if enc_token in joined:
            out["encoder"] = canonical
            break
    return out


# ---------------------------------------------------------------------------
# Filesystem walkers
# ---------------------------------------------------------------------------

def _ckpt_steps_in_dp_dir(d: Path) -> list[int]:
    """Return sorted step ints for *.ckpt files like '300.ckpt'."""
    out: list[int] = []
    for f in d.glob("*.ckpt"):
        m = re.match(r"(\d+)\.ckpt$", f.name)
        if m:
            out.append(int(m.group(1)))
    return sorted(out)


def walk_dp_checkpoints(root: Path) -> Iterator[RunRecord]:
    """Emit one RunRecord per DP ckpt dir under `root`.

    Each dir corresponds to one logical training run; the record reports the
    most-recent ckpt step and the full ckpt-paths list (comma-joined).
    """
    if not root.is_dir():
        return
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        steps = _ckpt_steps_in_dp_dir(d)
        if not steps:
            continue
        fields = parse_legacy_dp_dirname(d.name)
        latest_ckpt = str(d / f"{steps[-1]}.ckpt")
        rec = RunRecord(
            run_id=f"legacy:dp:{d.name}",
            phase="train",
            step=str(steps[-1]),
            ckpt_paths=latest_ckpt,
            status="done",
            notes=f"legacy DP ckpt dir; {len(steps)} step ckpts; min_step={steps[0]}",
            **fields,
        )
        yield rec


def walk_pi05_checkpoints(root: Path) -> Iterator[RunRecord]:
    """Emit one RunRecord per Pi0.5 (config, exp_name) pair under `root`.

    Layout: <root>/<config_name>/<exp_name>/<step>/ + optional wandb_id.txt.
    """
    if not root.is_dir():
        return
    for cfg_dir in sorted(root.iterdir()):
        if not cfg_dir.is_dir():
            continue
        for exp_dir in sorted(cfg_dir.iterdir()):
            if not exp_dir.is_dir():
                continue
            steps = sorted(
                int(p.name) for p in exp_dir.iterdir()
                if p.is_dir() and p.name.isdigit()
            )
            if not steps:
                continue
            wandb_id_file = exp_dir / "wandb_id.txt"
            wandb_id = wandb_id_file.read_text().strip() if wandb_id_file.is_file() else ""
            fields = _parse_pi05_config_name(cfg_dir.name)
            latest_ckpt = str(exp_dir / str(steps[-1]))
            rec = RunRecord(
                run_id=f"legacy:pi05:{cfg_dir.name}/{exp_dir.name}",
                model="pi05",
                phase="train",
                step=str(steps[-1]),
                ckpt_paths=latest_ckpt,
                status="done",
                wandb_url=(
                    f"https://wandb.ai/mikulrai-stanford-university/openpi-robofactory/runs/{wandb_id}"
                    if wandb_id else ""
                ),
                notes=(
                    f"legacy Pi0.5 ckpt dir; {len(steps)} step ckpts; "
                    f"min_step={steps[0]}; exp={exp_dir.name}"
                ),
                **fields,
            )
            yield rec


def _parse_pi05_config_name(name: str) -> dict[str, str]:
    """Best-effort field extraction from a Pi0.5 openpi config dir name."""
    out: dict[str, str] = {}
    if "wristcam" in name:
        out["dataset"] = "wc"
    elif "robofactory" in name:
        # default to workspace unless wristcam token present
        out["dataset"] = "ws"
    if "_pm_" in name or name.endswith("_pm"):
        out["task"] = "pm"
    if "_decent" in name:
        out["scheme"] = "decent"
    elif name.startswith("pi05_robofactory") and "decent" not in name:
        out["scheme"] = "cent"
    arm_match = re.search(r"_arm(\d)$", name)
    if arm_match:
        out["arm"] = arm_match.group(1)
    if "libero" in name:
        out["task"] = "libero"  # ablation marker
        out["category"] = "ablation"
    return out


# ---------------------------------------------------------------------------
# Wandb fetcher (optional — requires `wandb` package + login)
# ---------------------------------------------------------------------------

def fetch_wandb_runs(
    entity: str,
    projects: Iterable[str],
) -> Iterator[RunRecord]:
    """Fetch run summaries from wandb. Yields nothing if `wandb` is not
    importable or auth fails — caller decides how to handle that."""
    try:
        import wandb  # noqa: F401
        from wandb.apis.public import Api  # type: ignore
    except Exception:
        return
    api = Api()
    for project in projects:
        try:
            runs = api.runs(f"{entity}/{project}")
        except Exception:
            continue
        for r in runs:
            yield RunRecord(
                run_id=r.name or r.id,
                model="dp" if "diffusion" in project.lower() or "dp" in project.lower() else "pi05",
                wandb_url=r.url,
                status="done" if r.state == "finished" else r.state,
                created_utc=r.created_at,
                notes=f"wandb_id={r.id}; project={project}",
            )


# ---------------------------------------------------------------------------
# CSV writer
# ---------------------------------------------------------------------------

def write_manifest_csv(records: Iterable[RunRecord], path: Path) -> int:
    """Write `records` as CSV at `path`. Returns row count."""
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(MANIFEST_COLUMNS))
        w.writeheader()
        for r in records:
            w.writerow(r.as_row())
            n += 1
    return n


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=paths.MANIFEST_PATH)
    p.add_argument(
        "--source", choices=("filesystem", "wandb", "both"),
        default="filesystem",
        help="Data sources to include (default: filesystem only — no network)",
    )
    p.add_argument(
        "--dp-ckpt-root", type=Path,
        default=Path("/iris/u/mikulrai/checkpoints/RoboFactory"),
    )
    p.add_argument(
        "--pi05-ckpt-root", type=Path,
        default=Path("/iris/u/mikulrai/checkpoints/openpi"),
    )
    p.add_argument("--wandb-entity", default="mikulrai-stanford-university")
    p.add_argument(
        "--wandb-projects", default="openpi-robofactory",
        help="Comma-separated wandb project names",
    )
    p.add_argument("--dry-run", action="store_true", help="Print summary only; do not write")
    args = p.parse_args(argv)

    records: list[RunRecord] = []
    if args.source in ("filesystem", "both"):
        records.extend(walk_dp_checkpoints(args.dp_ckpt_root))
        records.extend(walk_pi05_checkpoints(args.pi05_ckpt_root))
    if args.source in ("wandb", "both"):
        projects = [s.strip() for s in args.wandb_projects.split(",") if s.strip()]
        records.extend(fetch_wandb_runs(args.wandb_entity, projects))

    print(f"manifest: collected {len(records)} record(s) from source={args.source}")
    by_model: dict[str, int] = {}
    for r in records:
        by_model[r.model or "?"] = by_model.get(r.model or "?", 0) + 1
    for k, v in sorted(by_model.items()):
        print(f"  model={k!r:>6}: {v}")

    if args.dry_run:
        print(f"manifest: dry-run — would write {args.output}")
        return 0

    n = write_manifest_csv(records, args.output)
    print(f"manifest: wrote {n} rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
