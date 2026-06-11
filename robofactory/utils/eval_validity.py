"""Eval validity guards, vocab-checked prompts, and provenance v2 (solution_E6.md §PR6).

Three pure, GPU-free helper groups every eval driver runs so a garbage eval can
never again be silently counted as a result:

1. VALIDITY GUARD (``classify_validity`` / ``finalize_validity``)
   Counts episodes with ``steps == -1`` or a non-empty ``error`` field. If more
   than ``INVALID_FRACTION_THRESHOLD`` (5%) of episodes are invalid the run is
   declared ``valid=False`` with an ``invalid_reason``, the output JSON is renamed
   ``*_INVALID.json``, the process exits 2, and any wandb run is tagged ``invalid``
   (the 06-05 o4ssi4ek garbage synced as a normal run — audit_A9 §4a). An explicit
   signature match for ``IndexError('index 15 is out of bounds for axis 0 with
   size 8')`` maps to the "wrong-policy / port-collision" diagnosis (see PR2
   handshake).

2. PROMPT GUARD (``load_prompt_vocab`` / ``validate_prompt``)
   The old ``DEFAULT_PROMPT = "stack the three cubes using three robot arms"`` (a
   TSC string) was silently used for EVERY task (digest B item B4). ``--prompt`` is
   now required; it is validated against the task's prompts JSON
   (``examples/robofactory/*_prompts.json`` / ``camera_mappings/*_prompts.json``)
   or, for sidecar-trained ckpts, the subtask vocab in ``lb_subtask_index.npz``. An
   out-of-vocabulary prompt hard-fails and prints the vocab. ``--allow-oov-prompt``
   overrides the check and records itself in the result JSON (needed for the E1
   prompt-swap probe).

3. PROVENANCE v2 (``build_provenance``)
   Every result JSON records hostname, GPU name, git sha + dirty flag of BOTH repos
   (RoboFactory + openpi), shader_pack + enable_shadow + resolved sim_backend, seed
   pool name + sha (from PR4), prompt string(s), max env steps + chunk config,
   server metadata (pi0.5, from PR2) or ckpt path + md5 + zarr repo_id (DP), and the
   protocol tag ``eval_protocol = "v2"``.
"""
from __future__ import annotations

import hashlib
import json
import os
import socket
import subprocess
import sys
import warnings
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence

# --------------------------------------------------------------------------- repo roots

ROBOFACTORY_REPO = "/iris/u/mikulrai/projects/RoboFactory"
OPENPI_REPO = "/iris/u/mikulrai/projects/openpi"

EVAL_PROTOCOL = "v2"

# ===========================================================================
# 1. VALIDITY GUARD
# ===========================================================================

# More than this fraction of invalid episodes -> the whole run is invalid.
INVALID_FRACTION_THRESHOLD = 0.05

# Exit code for an invalid eval run (distinct from 0 success / 1 generic crash /
# 3 server-identity mismatch). Launchers/CI key on 2 == "ran but produced garbage".
INVALID_EXIT_CODE = 2

# The wrong-policy / port-collision signature: a 2-arm (size-8) action chunk
# served behind a port a >=2-arm client expected, so it indexes element 15 of an
# 8-element axis (see memory/project_pi05_eval_port_collision and PR2 handshake).
PORT_COLLISION_SIGNATURE = "index 15 is out of bounds for axis 0 with size 8"
PORT_COLLISION_REASON = (
    "wrong-policy/port-collision signature (see PR2 handshake): "
    "an episode raised IndexError('index 15 is out of bounds for axis 0 with size 8'), "
    "which means a smaller-action-dim policy was served behind the expected port. "
    "Free the ports at runtime and verify the server-identity handshake."
)

INVALID_TAG = "invalid"


def _episode_is_invalid(ep: Mapping) -> bool:
    """True iff an episode dict signals a failed/garbage run (not a clean 0-success).

    Two signals, matching what the drivers already write into a result row:
      - ``steps == -1`` (the exception path sets this in every driver), or
      - a non-empty ``error`` field.
    A legitimately-failed episode (success=0, steps>=0, no error) is VALID."""
    if int(ep.get("steps", 0)) == -1:
        return True
    err = ep.get("error")
    return bool(err)  # non-empty / non-None error string


def _has_port_collision_signature(results: Iterable[Mapping]) -> bool:
    for ep in results:
        err = ep.get("error")
        if err and PORT_COLLISION_SIGNATURE in str(err):
            return True
    return False


def classify_validity(
    results: Sequence[Mapping],
    threshold: float = INVALID_FRACTION_THRESHOLD,
) -> dict:
    """Classify a finished run as valid / invalid from its per-episode results.

    Returns a JSON-safe dict:
      ``{valid, invalid_reason, n_episodes, n_invalid, invalid_fraction, threshold}``.
    ``valid`` is False when ``n_invalid / n_episodes > threshold`` (or there are
    zero episodes). ``invalid_reason`` is the port-collision diagnosis if any
    episode carries that signature, else a generic count summary; it is ``None``
    when the run is valid."""
    n = len(results)
    n_invalid = sum(1 for ep in results if _episode_is_invalid(ep))
    frac = (n_invalid / n) if n else 1.0
    valid = (n > 0) and (frac <= threshold)
    reason: Optional[str] = None
    if not valid:
        if _has_port_collision_signature(results):
            reason = PORT_COLLISION_REASON
        elif n == 0:
            reason = "no episodes ran (empty results)"
        else:
            reason = (
                f"{n_invalid}/{n} episodes invalid "
                f"(steps==-1 or nonempty error) = {100.0 * frac:.1f}% "
                f"> {100.0 * threshold:.0f}% threshold"
            )
    return {
        "valid": bool(valid),
        "invalid_reason": reason,
        "n_episodes": n,
        "n_invalid": n_invalid,
        "invalid_fraction": frac,
        "threshold": threshold,
    }


def invalid_output_path(out_file) -> Path:
    """``…/foo.json`` -> ``…/foo_INVALID.json`` (also handles ``.jsonl``; idempotent)."""
    p = Path(out_file)
    if p.stem.endswith("_INVALID"):
        return p
    if p.suffix in (".json", ".jsonl"):
        return p.with_name(p.stem + "_INVALID" + p.suffix)
    return p.with_name(p.name + "_INVALID")


def finalize_validity(
    validity: Mapping,
    out_file,
    *,
    wandb_run=None,
    exit_on_invalid: bool = True,
) -> Path:
    """Apply the validity verdict: rename the output, tag wandb, and exit 2 if invalid.

    Args:
        validity: the dict from ``classify_validity`` (already merged into the JSON
            the caller wrote at ``out_file``).
        out_file: the path the caller already wrote the result JSON to.
        wandb_run: optional ``WandbRun`` (or any object exposing ``.run``) — tagged
            ``invalid`` when the run is invalid so a garbage eval is never mistaken
            for a normal wandb run.
        exit_on_invalid: when True (default) and the run is invalid, ``sys.exit(2)``
            AFTER renaming + tagging. Set False in tests.

    Returns the (possibly renamed) path the result JSON lives at."""
    final_path = Path(out_file)
    if validity.get("valid", True):
        return final_path

    # Rename foo.json -> foo_INVALID.json so a downstream grep never counts it.
    renamed = invalid_output_path(out_file)
    try:
        if final_path.exists() and renamed != final_path:
            final_path.rename(renamed)
            final_path = renamed
    except OSError as e:  # pragma: no cover - filesystem race
        warnings.warn(f"[validity] could not rename {final_path} -> {renamed}: {e}")

    # Tag the wandb run 'invalid' (best-effort; never raise from the tagger).
    _tag_wandb_invalid(wandb_run)

    print(
        f"[validity] RUN INVALID: {validity.get('invalid_reason')}\n"
        f"[validity] output renamed to {final_path}; exiting {INVALID_EXIT_CODE}",
        file=sys.stderr,
        flush=True,
    )
    if exit_on_invalid:
        sys.exit(INVALID_EXIT_CODE)
    return final_path


def _tag_wandb_invalid(wandb_run) -> None:
    """Best-effort: add the ``invalid`` tag to a live wandb run. Never raises."""
    if wandb_run is None:
        return
    run = getattr(wandb_run, "run", wandb_run)
    if run is None:
        return
    try:
        existing = list(getattr(run, "tags", []) or [])
        if INVALID_TAG not in existing:
            run.tags = existing + [INVALID_TAG]
    except Exception as e:  # pragma: no cover - wandb internals
        warnings.warn(f"[validity] could not tag wandb run invalid: {e}")


# ===========================================================================
# 2. PROMPT GUARD
# ===========================================================================

# Where prompt vocabularies live. Both task-prompt JSONs and the LB subtask npz
# are searched. These are the openpi-side data the train pipeline used.
_PROMPTS_DIRS = (
    Path(OPENPI_REPO) / "examples" / "robofactory" / "camera_mappings",
    Path(OPENPI_REPO) / "examples" / "robofactory",
)
LB_SUBTASK_NPZ = Path(
    "/iris/u/mikulrai/data/RoboFactory/lerobot/lb_subtask_sidecar/lb_subtask_index.npz"
)


class PromptOutOfVocabError(ValueError):
    """Raised when --prompt is not in the resolved task / subtask vocabulary."""


def _read_prompts_json(path: Path) -> list[str]:
    """Read a prompts JSON. Accepts a flat list of strings, or a dict whose values
    are strings/lists of strings (defensive — current files are flat lists)."""
    data = json.loads(Path(path).read_text())
    out: list[str] = []
    if isinstance(data, list):
        out = [str(x) for x in data]
    elif isinstance(data, dict):
        for v in data.values():
            if isinstance(v, str):
                out.append(v)
            elif isinstance(v, (list, tuple)):
                out.extend(str(x) for x in v)
    return out


def _load_subtask_vocab(npz_path: Path = LB_SUBTASK_NPZ) -> list[str]:
    """Return the LB sidecar subtask vocab (the verbalized subtask strings the
    sidecar-trained LL policies were conditioned on)."""
    import numpy as np

    d = np.load(npz_path, allow_pickle=True)
    if "vocab" in d:
        return [str(x) for x in d["vocab"].tolist()]
    # Fall back to the union of the per-step left/right text columns.
    cols: list[str] = []
    for key in ("left_text", "right_text"):
        if key in d:
            cols.extend(str(x) for x in d[key].tolist())
    return sorted(set(cols))


def resolve_prompts_json(prompts_json: Optional[str], task: Optional[str]) -> Optional[Path]:
    """Find the prompts JSON for this run.

    Priority: an explicit ``--prompts-json`` path > a ``<task>_prompts.json`` derived
    from a camera-mapping file / task name. Returns the resolved Path or None."""
    if prompts_json:
        p = Path(prompts_json)
        if p.exists():
            return p
        raise FileNotFoundError(f"--prompts-json not found: {prompts_json}")
    if not task:
        return None
    # Map a camera-mapping path or env-id to a prompts-json stem.
    stem = Path(task).stem
    # strip a trailing -rf env-id suffix and known camera-family suffixes
    stem = stem.replace("-rf", "")
    for suffix in ("_wristcam", "_union"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
    # CamelCase env-id (ThreeRobotsStackCube) -> we cannot map reliably; only try
    # the snake_case forms that exist as files.
    candidates = [stem, _camel_to_snake(stem)]
    for cand in candidates:
        for d in _PROMPTS_DIRS:
            f = d / f"{cand}_prompts.json"
            if f.exists():
                return f
    return None


def _camel_to_snake(name: str) -> str:
    import re

    s = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s)
    return s.lower()


def load_prompt_vocab(
    *,
    prompts_json: Optional[str] = None,
    task: Optional[str] = None,
    use_subtask_vocab: bool = False,
    subtask_npz: Optional[str] = None,
) -> tuple[list[str], str]:
    """Resolve the allowed-prompt vocabulary for a run.

    Returns ``(vocab, source)`` where ``source`` names where the vocab came from
    (the prompts-json path, the subtask npz path, or ``"<none>"`` when no vocab
    could be located — in which case the caller cannot validate and should warn)."""
    if use_subtask_vocab or subtask_npz:
        npz = Path(subtask_npz) if subtask_npz else LB_SUBTASK_NPZ
        return _load_subtask_vocab(npz), str(npz)
    pj = resolve_prompts_json(prompts_json, task)
    if pj is not None:
        return _read_prompts_json(pj), str(pj)
    return [], "<none>"


def validate_prompt(
    prompt: Optional[str],
    vocab: Sequence[str],
    *,
    allow_oov: bool = False,
    vocab_source: str = "",
) -> dict:
    """Validate ``prompt`` against ``vocab``. Hard-fail (raise) on OOV unless allowed.

    Returns a JSON-safe provenance dict:
      ``{prompt, prompt_in_vocab, allow_oov_prompt, vocab_source, vocab_size}``.
    A missing/empty prompt always raises (``--prompt`` is required). An empty vocab
    (no prompts file located) cannot validate — it warns and records
    ``prompt_in_vocab = None``."""
    if not prompt:
        raise ValueError(
            "--prompt is REQUIRED (PR6): the old DEFAULT_PROMPT='stack the three "
            "cubes using three robot arms' was a TSC string silently used for every "
            "task. Pass --prompt explicitly."
        )
    vocab_list = [str(v) for v in vocab]
    in_vocab: Optional[bool]
    if not vocab_list:
        warnings.warn(
            f"[prompt] no prompt vocabulary located ({vocab_source!r}); cannot "
            "validate --prompt. Pass --prompts-json or --subtask-vocab to enable "
            "the OOV guard."
        )
        in_vocab = None
    else:
        in_vocab = prompt in vocab_list
        if not in_vocab and not allow_oov:
            raise PromptOutOfVocabError(
                f"--prompt {prompt!r} is OUT OF VOCABULARY for {vocab_source}.\n"
                f"Allowed prompts ({len(vocab_list)}):\n"
                + "\n".join(f"  - {v!r}" for v in vocab_list)
                + "\n(Pass --allow-oov-prompt to override for a deliberate prompt-swap "
                "probe; that override is recorded in the result JSON.)"
            )
        if not in_vocab and allow_oov:
            warnings.warn(
                f"[prompt] --prompt {prompt!r} is OOV for {vocab_source} but "
                "--allow-oov-prompt is set; proceeding and recording the override."
            )
    return {
        "prompt": prompt,
        "prompt_in_vocab": in_vocab,
        "allow_oov_prompt": bool(allow_oov),
        "vocab_source": vocab_source,
        "vocab_size": len(vocab_list),
    }


# ===========================================================================
# 3. PROVENANCE v2
# ===========================================================================


def git_provenance(repo_dir: str) -> dict:
    """``{sha, dirty}`` for a git repo. sha='unknown' / dirty=None if unavailable."""
    out = {"sha": "unknown", "dirty": None}
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_dir, stderr=subprocess.DEVNULL
        ).decode().strip()
        out["sha"] = sha
    except Exception:
        return out
    try:
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=repo_dir, stderr=subprocess.DEVNULL
        ).decode()
        out["dirty"] = bool(status.strip())
    except Exception:
        out["dirty"] = None
    return out


def gpu_name() -> Optional[str]:
    """Best-effort GPU model name. Tries torch first, then nvidia-smi, else None."""
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:
        pass
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            stderr=subprocess.DEVNULL,
        ).decode().strip().splitlines()
        if out:
            return out[0].strip()
    except Exception:
        pass
    return None


def file_md5(path: Optional[str]) -> Optional[str]:
    """md5 of a file (DP ckpt identity). None if path missing/unreadable."""
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    try:
        h = hashlib.md5()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def build_provenance(
    *,
    shader_pack: str = "default",
    enable_shadow: Optional[bool] = None,
    sim_backend: Optional[str] = None,
    seed_provenance: Optional[Mapping] = None,
    prompts: Optional[Sequence[str]] = None,
    prompt_validation: Optional[Mapping] = None,
    max_env_steps: Optional[int] = None,
    chunk_config: Optional[Mapping] = None,
    server_metadata: Optional[Mapping] = None,
    ckpt_paths: Optional[Sequence[str]] = None,
    zarr_repo_id: Optional[str] = None,
    extra: Optional[Mapping] = None,
) -> dict:
    """Assemble the protocol-v2 provenance block embedded in every result JSON.

    Records: hostname, GPU name, git sha+dirty of BOTH repos, shader_pack +
    enable_shadow + resolved sim_backend, seed pool name+sha (PR4), prompt
    string(s) + OOV verdict, max env steps + chunk config, server metadata (pi0.5,
    PR2) OR ckpt path+md5+zarr repo_id (DP), and ``eval_protocol = "v2"``."""
    prov: dict = {
        "eval_protocol": EVAL_PROTOCOL,
        "hostname": socket.gethostname(),
        "gpu_name": gpu_name(),
        "git": {
            "robofactory": git_provenance(ROBOFACTORY_REPO),
            "openpi": git_provenance(OPENPI_REPO),
        },
        "shader_pack": shader_pack,
        "enable_shadow": enable_shadow,
        "sim_backend": sim_backend,
        "seed_provenance": dict(seed_provenance) if seed_provenance else None,
        "prompts": list(prompts) if prompts is not None else None,
        "prompt_validation": dict(prompt_validation) if prompt_validation else None,
        "max_env_steps": max_env_steps,
        "chunk_config": dict(chunk_config) if chunk_config else None,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    if server_metadata is not None:
        prov["server_metadata"] = dict(server_metadata)
    if ckpt_paths is not None:
        prov["ckpt_paths"] = list(ckpt_paths)
        prov["ckpt_md5"] = {str(p): file_md5(p) for p in ckpt_paths}
    if zarr_repo_id is not None:
        prov["zarr_repo_id"] = zarr_repo_id
    if extra:
        prov.update(dict(extra))
    return prov


__all__ = [
    # validity
    "INVALID_FRACTION_THRESHOLD",
    "INVALID_EXIT_CODE",
    "INVALID_TAG",
    "PORT_COLLISION_SIGNATURE",
    "PORT_COLLISION_REASON",
    "classify_validity",
    "finalize_validity",
    "invalid_output_path",
    # prompt
    "PromptOutOfVocabError",
    "load_prompt_vocab",
    "validate_prompt",
    "resolve_prompts_json",
    # provenance
    "EVAL_PROTOCOL",
    "build_provenance",
    "git_provenance",
    "gpu_name",
    "file_md5",
]
