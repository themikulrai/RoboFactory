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
import json
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
            fields = _parse_pi05_config_name(cfg_dir.name, exp_name=exp_dir.name)
            # If the parser already produced a notes string, fold it into the
            # final notes below; otherwise use only the per-dir summary.
            parser_notes = fields.pop("notes", "")
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
                    + (f"; {parser_notes}" if parser_notes else "")
                ),
                **fields,
            )
            yield rec


def _parse_pi05_config_name(
    name: str,
    exp_name: Optional[str] = None,
) -> dict[str, str]:
    """Best-effort field extraction from a Pi0.5 openpi config dir name.

    `name` is the top-level config dir (e.g. ``pi05_robofactory_pm_lora_finetune``).
    `exp_name` is the inner directory (e.g. ``pi05_pm_d1_v1``) — when provided
    it can disambiguate the task in cases where the config dir omits it.
    """
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

    # Task disambiguation from exp_name when the config dir didn't reveal one.
    if "task" not in out and exp_name:
        for tok, task in (("_pm_", "pm"), ("_2sc_", "2sc"),
                          ("_tsc_", "tsc"), ("_lp_", "lp")):
            if tok in f"_{exp_name}_":
                out["task"] = task
                break

    # Codebase convention (2026-05): every Pi0.5 decent finetune we've shipped
    # is TSC. If task is still blank for a decent row, assume tsc and record
    # the assumption in `notes` so a future 2sc/lp run is easy to flag.
    if "task" not in out and out.get("scheme") == "decent":
        out["task"] = "tsc"
        out["notes"] = (
            "task=tsc inferred from scheme=decent (codebase convention 2026-05); "
            "verify manually if a 2sc/lp decent run lands"
        )
    return out


# ---------------------------------------------------------------------------
# JSONL fallback for slurm-killed runs (workflow improvement #6)
# ---------------------------------------------------------------------------

def _eval_sr_from_jsonl(config: dict) -> tuple[Optional[float], Optional[int]]:
    """Recover (success_rate, n_total) from the eval JSONL on disk.

    The eval_dp / eval_multi_dp drivers bake `jsonl_path` into the manifest
    they pass to `wandb.init(config=...)` (see eval_dp.py:348). When SLURM
    kills the job before `wandb.finish()` runs, the wandb summary stays
    empty but the JSONL on disk is intact. This helper reads it.

    Layout:
      line 1     -- {"kind": "manifest", ...}
      lines 2..N -- {"kind": "episode", "success": 0/1, ...}
      last line  -- {"kind": "summary", "success_rate": ..., "n_total": ...}
                    (absent if killed mid-eval)

    Strategy:
      1. If a `summary` line exists, return its fields verbatim.
      2. Otherwise, count episode rows + their `success` field and
         compute SR + n_total ourselves.
    """
    path_str = config.get("jsonl_path") or config.get("results_json_path")
    if not path_str:
        return None, None
    path = Path(str(path_str))
    if not path.is_file():
        return None, None

    sr: Optional[float] = None
    n: Optional[int] = None
    n_succ = 0
    n_total = 0
    try:
        with path.open() as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                kind = row.get("kind")
                if kind == "summary":
                    if "success_rate" in row:
                        sr = float(row["success_rate"])
                    for nk in ("n_total", "n_episodes", "n"):
                        if nk in row:
                            n = int(row[nk])
                            break
                elif kind == "episode":
                    n_total += 1
                    if int(row.get("success", 0)):
                        n_succ += 1
    except OSError:
        return None, None

    if sr is not None:
        return sr, n
    if n_total > 0:
        return n_succ / n_total, n_total
    return None, None


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
    try:
        api = Api()
    except Exception:
        return
    for project in projects:
        try:
            runs = api.runs(f"{entity}/{project}")
        except Exception:
            continue
        for r in runs:
            try:
                yield wandb_run_to_record(r, project=project)
            except Exception:
                # Defensive: a single malformed wandb run must not nuke the
                # whole manifest pass. Skip and continue.
                continue


def wandb_run_to_record(r, *, project: str) -> "RunRecord":
    """Convert a wandb Run object (or duck-typed equivalent) into a RunRecord.

    Factored out of fetch_wandb_runs so tests can build a fake run object
    without spinning up wandb.Api().
    """
    name = getattr(r, "name", None) or getattr(r, "id", "") or ""
    state = getattr(r, "state", "") or ""
    config = getattr(r, "config", {}) or {}
    summary = getattr(r, "summary", {}) or {}
    tags = list(getattr(r, "tags", []) or [])

    # Status map: wandb states -> manifest statuses.
    state_map = {
        "finished": "done",
        "failed": "failed",
        "crashed": "failed",
        "killed": "failed",
        "running": "running",
        "queued": "queued",
        "preempted": "queued",
    }
    status = state_map.get(state, state or "unknown")

    # Best-effort field extraction. Prefer canonical run_id parser; fall
    # back to legacy DP / pi05 config-name heuristics.
    parsed = parse_canonical_run_id(name) or {}
    fields: dict[str, str] = {}
    for k in (
        "model", "dataset", "task", "encoder", "scheme", "arm",
        "phase", "step", "seed", "git_sha",
    ):
        v = parsed.get(k, "")
        if v:
            fields[k] = v

    # Project hint when canonical parser failed.
    if not fields.get("model"):
        plow = project.lower()
        if "diffusion" in plow or plow.startswith("dp"):
            fields["model"] = "dp"
        elif "openpi" in plow or "pi05" in plow:
            fields["model"] = "pi05"

    # Common config keys that might give us extra info.
    for cfg_key, fld in (
        ("seed", "seed"),
        ("git_sha", "git_sha"),
        ("slurm_job_id", "slurm_id"),
        ("SLURM_JOB_ID", "slurm_id"),
    ):
        if cfg_key in config and not fields.get(fld):
            fields[fld] = str(config[cfg_key])

    # Eval summary metrics. eval_decent_pi05.py + eval_multi_dp.py both log
    # final SR under `summary/success_rate` (literal slash in key name); accept
    # those plus any flat `success_rate` / `eval/success_rate` aliases.
    eval_sr = ""
    eval_n = ""
    for sk in ("summary/success_rate", "eval/success_rate", "success_rate", "eval_sr"):
        if sk in summary:
            eval_sr = str(summary[sk])
            break
    for nk in ("summary/n_total", "summary/n_episodes", "eval/n", "n_episodes", "eval_n"):
        if nk in summary:
            eval_n = str(summary[nk])
            break

    # JSONL fallback (workflow improvement #6): when SLURM kills wandb mid-eval
    # (e.g. timelimit hit before `wandb.finish()`), `summary/*` is empty even
    # though the on-disk JSONL has every episode row. Recover by tailing the
    # JSONL's last summary line, or — if the summary line itself is missing
    # because the kill landed mid-eval — by computing SR from the episode
    # rows on the fly. Path comes from the eval_dp / eval_multi_dp manifest
    # the eval driver baked into wandb config at init time.
    if not eval_sr:
        sr_jsonl, n_jsonl = _eval_sr_from_jsonl(config)
        if sr_jsonl is not None:
            eval_sr = str(sr_jsonl)
            if not eval_n and n_jsonl is not None:
                eval_n = str(n_jsonl)

    # Phase fallback: if the run name lacks a phase token but config or
    # tags indicate an eval, mark it eval.
    if not fields.get("phase"):
        if any(t.lower() == "eval" for t in tags) or "eval" in name.lower():
            fields["phase"] = "eval"
        elif any(t.lower() == "train" for t in tags) or "train" in name.lower():
            fields["phase"] = "train"

    note_bits = [f"wandb_id={getattr(r, 'id', '')}", f"project={project}"]
    if tags:
        note_bits.append("tags=" + ",".join(tags))

    return RunRecord(
        run_id=name,
        wandb_url=getattr(r, "url", "") or "",
        status=status,
        created_utc=str(getattr(r, "created_at", "") or ""),
        eval_sr=eval_sr,
        eval_n=eval_n,
        notes="; ".join(note_bits),
        **fields,
    )


# ---------------------------------------------------------------------------
# CkptResolver-backed scanner + merge logic for generate_manifest_csv()
# ---------------------------------------------------------------------------


def _ckpt_entry_to_record(entry) -> RunRecord:
    """Build a RunRecord from one CkptEntry (per ckpt_index.jsonl row).

    The framework token from CkptResolver maps directly to manifest.model.
    """
    framework = getattr(entry, "framework", "")
    identifier = getattr(entry, "identifier", "")
    epoch = getattr(entry, "epoch", "")
    path_str = getattr(entry, "path", "")
    mtime = getattr(entry, "mtime_unix", None)

    fields: dict[str, str] = {"model": framework if framework in _VALID_MODELS else ""}
    if framework == "dp":
        fields.update(parse_legacy_dp_dirname(identifier.split(":", 1)[0]))
    elif framework == "pi05":
        # identifier is "<config>/<exp>" for pi05 entries
        if "/" in identifier:
            cfg_name, exp_name = identifier.split("/", 1)
        else:
            cfg_name, exp_name = identifier, ""
        fields.update(_parse_pi05_config_name(cfg_name, exp_name=exp_name))

    parser_notes = fields.pop("notes", "")
    fields.setdefault("model", framework)

    created = ""
    if mtime is not None:
        try:
            created = _dt.datetime.utcfromtimestamp(float(mtime)).isoformat() + "Z"
        except (OverflowError, OSError, ValueError):
            created = ""

    return RunRecord(
        run_id=f"legacy:{framework}:{identifier}",
        phase="train",
        step=str(epoch),
        ckpt_paths=path_str,
        status="done",
        created_utc=created,
        notes=("ckpt_index entry" + (f"; {parser_notes}" if parser_notes else "")),
        **fields,
    )


def _records_from_ckpt_index(index_path: Path) -> Iterator[RunRecord]:
    """Yield RunRecords from the ckpt_index.jsonl produced by CkptResolver."""
    try:
        from robofactory.utils.ckpt_resolver import CkptResolver
    except Exception:
        return
    if not Path(index_path).is_file():
        return
    res = CkptResolver.load(Path(index_path))
    # Group by (framework, identifier) so we emit one row per logical run
    # with the latest ckpt path; track ckpt count for notes.
    by_run: dict[tuple[str, str], list] = {}
    for e in res:
        by_run.setdefault((e.framework, e.identifier), []).append(e)
    for (_fw, _ident), entries in by_run.items():
        latest = max(entries, key=lambda x: (x.epoch, x.mtime_unix))
        rec = _ckpt_entry_to_record(latest)
        rec.notes = (
            (rec.notes + "; " if rec.notes else "")
            + f"{len(entries)} ckpt rows in index; min_epoch={min(e.epoch for e in entries)}"
        )
        yield rec


def _walk_run_root(run_root: Path) -> Iterator[RunRecord]:
    """Walk RUN_ROOT/<model>/<dataset>/<task>/<run_id>/ for canonical runs.

    Layout per plan v2 C1: each canonical run gets its own subdir under
    RUN_ROOT. Empty in early phases — caller treats zero rows as fine.
    """
    if not Path(run_root).is_dir():
        return
    for model_dir in Path(run_root).iterdir():
        if not model_dir.is_dir() or model_dir.name not in _VALID_MODELS:
            continue
        for ds_dir in model_dir.iterdir():
            if not ds_dir.is_dir():
                continue
            for task_dir in ds_dir.iterdir():
                if not task_dir.is_dir():
                    continue
                for run_dir in task_dir.iterdir():
                    if not run_dir.is_dir():
                        continue
                    parsed = parse_canonical_run_id(run_dir.name) or {}
                    rec = RunRecord(
                        run_id=run_dir.name,
                        status="unknown",
                        notes=f"run_dir={run_dir}",
                        **{k: v for k, v in parsed.items() if k in {
                            "model", "dataset", "task", "encoder", "scheme",
                            "arm", "phase", "step", "seed", "git_sha",
                        }},
                    )
                    yield rec


def _merge_records(
    wandb_recs: Iterable[RunRecord],
    ckpt_recs: Iterable[RunRecord],
    runroot_recs: Iterable[RunRecord],
) -> list[RunRecord]:
    """Merge the three streams into a deduplicated list.

    Strategy:
      * primary key is run_id when canonical; for legacy ckpt rows it is
        ``legacy:<framework>:<identifier>``.
      * a wandb record whose run_id matches a ckpt record by exact run_id
        OR whose notes contain the ckpt identifier merges the two — wandb
        contributes status/url/eval metrics, ckpt contributes ckpt_paths.
      * un-matched records pass through with their own run_ids.
    """
    wandb_list = list(wandb_recs)
    ckpt_list = list(ckpt_recs)
    runroot_list = list(runroot_recs)

    # Index ckpts by run_id and by short identifier.
    ckpt_by_id: dict[str, RunRecord] = {r.run_id: r for r in ckpt_list}
    used_ckpt_ids: set[str] = set()

    merged: list[RunRecord] = []

    for wrec in wandb_list:
        match = ckpt_by_id.get(wrec.run_id)
        if match is None:
            # Try to match by structural fields (model+dataset+task+arm+seed).
            for cid, crec in ckpt_by_id.items():
                if cid in used_ckpt_ids:
                    continue
                if (
                    wrec.model and wrec.model == crec.model
                    and wrec.task and wrec.task == crec.task
                    and wrec.scheme and wrec.scheme == crec.scheme
                    and wrec.arm == crec.arm
                    and wrec.seed and wrec.seed == crec.seed
                ):
                    match = crec
                    break
        if match is not None:
            used_ckpt_ids.add(match.run_id)
            merged.append(_combine(wrec, match))
        else:
            merged.append(wrec)

    for cid, crec in ckpt_by_id.items():
        if cid not in used_ckpt_ids:
            merged.append(crec)

    # Run-root walk emits canonical-runid rows; treat as authoritative for
    # any run_id not already present.
    seen_ids = {r.run_id for r in merged}
    for rrec in runroot_list:
        if rrec.run_id not in seen_ids:
            merged.append(rrec)
            seen_ids.add(rrec.run_id)

    return merged


def _combine(wandb_rec: RunRecord, ckpt_rec: RunRecord) -> RunRecord:
    """Field-wise merge: prefer wandb for metadata, ckpt for paths."""
    out = dataclasses.replace(wandb_rec)
    # ckpt_paths and step come from ckpt side
    if ckpt_rec.ckpt_paths and not out.ckpt_paths:
        out.ckpt_paths = ckpt_rec.ckpt_paths
    if ckpt_rec.step and not out.step:
        out.step = ckpt_rec.step
    # fill structural fields the wandb parser missed
    for f in ("model", "dataset", "task", "encoder", "scheme", "arm"):
        if not getattr(out, f) and getattr(ckpt_rec, f):
            setattr(out, f, getattr(ckpt_rec, f))
    if ckpt_rec.created_utc and not out.created_utc:
        out.created_utc = ckpt_rec.created_utc
    if ckpt_rec.notes:
        out.notes = (out.notes + "; " if out.notes else "") + ckpt_rec.notes
    return out


# ---------------------------------------------------------------------------
# Top-level generator (additive API used by `python -m robofactory.utils.manifest_csv`)
# ---------------------------------------------------------------------------

def generate_manifest_csv(
    out_path: Optional[Path] = None,
    *,
    fetch_wandb: bool = True,
    wandb_entity: str = "mikulrai-stanford-university",
    wandb_projects: tuple[str, ...] = ("diffusion-robofactory", "openpi-robofactory"),
    ckpt_index_path: Optional[Path] = None,
    run_root: Optional[Path] = None,
) -> int:
    """Regenerate the 23-column manifest CSV from wandb + ckpt index + runs/.

    Returns the number of data rows written. Wraps the wandb call in a
    try/except so an offline/down API never crashes the manifest build.
    """
    if out_path is None:
        out_path = paths.MANIFEST_PATH
    if ckpt_index_path is None:
        ckpt_index_path = paths.CKPT_INDEX_PATH
    if run_root is None:
        run_root = paths.RUN_ROOT

    wandb_recs: list[RunRecord] = []
    if fetch_wandb:
        try:
            wandb_recs = list(fetch_wandb_runs(wandb_entity, list(wandb_projects)))
        except Exception as e:
            # Soft-fail: proceed with whatever filesystem data we have.
            print(f"manifest_csv: wandb fetch failed ({e!r}); continuing without wandb rows")
            wandb_recs = []

    ckpt_recs = list(_records_from_ckpt_index(Path(ckpt_index_path)))
    runroot_recs = list(_walk_run_root(Path(run_root)))

    merged = _merge_records(wandb_recs, ckpt_recs, runroot_recs)
    n = write_manifest_csv(merged, Path(out_path))
    return n


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
