"""Shared eval context managers used by every `eval_*.py` driver.

Three primitives:

- `assert_wandb_live()` — refuses to proceed if wandb cannot reach the cloud
  (`WANDB_MODE != "online"`, `WANDB_DISABLED` set, or no API key visible).
- `WandbRun(...)` — context manager that init/finish-es a wandb run.
  `enabled=False` short-circuits to a no-op so existing `--wandb` CLI flags work.
- `VideoRecorder(...)` — context manager that returns video paths only for the
  first N seeds (default 3) unless `all_seeds=True`.

Usage:

    with WandbRun(enabled=args.wandb, project="...", job_type="eval",
                  name="...", tags=[...], config={...}) as run:
        with VideoRecorder(record_dir, max_recorded=3) as videos:
            for seed_idx, seed in enumerate(seeds):
                video_path = videos.video_path_for(seed_idx, seed)
                metrics = run_episode(..., video_path=video_path)
                run.log_episode(seed_idx, **metrics)
            run.log_summary(success_rate=sr, n_episodes=n)
"""
from __future__ import annotations

import os
from typing import Optional


def assert_wandb_live() -> None:
    """Crash if wandb is configured to run offline / disabled / no-credentials.

    The `eval_decent_pi05.py` lapse (~30 GPU-hours of eval with no wandb run) is
    the canonical motivating incident. By calling this from `WandbRun.__enter__`
    every driver inherits the same guard.
    """
    mode = os.environ.get("WANDB_MODE", "online")
    if mode != "online":
        raise RuntimeError(
            f"WANDB_MODE='{mode}' but eval drivers require 'online'. "
            "Unset WANDB_MODE or set WANDB_MODE=online and retry."
        )
    if os.environ.get("WANDB_DISABLED"):
        raise RuntimeError(
            "WANDB_DISABLED is set; eval drivers must log to wandb. "
            "Unset WANDB_DISABLED and retry."
        )
    if not os.environ.get("WANDB_API_KEY") and not os.path.exists(
        os.path.expanduser("~/.netrc")
    ):
        raise RuntimeError(
            "No WANDB_API_KEY and no ~/.netrc entry. Run `wandb login` first."
        )


# ---------------------------------------------------------------------------
# Workflow improvement #3 — auto-derive wandb tags from run name
# ---------------------------------------------------------------------------

# Tokens we want as wandb tags whenever they appear in the run name. Order
# is fixed so tag sets sort consistently across runs. Token matching is
# case-insensitive and word-boundary aware (so "in1k" doesn't match "in1k_").
# Add tokens here as new training axes appear; nothing else needs to change.
_TAG_TOKENS = (
    # encoder choice
    "in1k", "freezeenc", "dinos", "dino", "r3m", "lora",
    # task
    "pm", "tsc", "2sc", "lp", "libero",
    # dataset variant
    "d1", "d2",
    # camera family
    "ws", "wc", "workspace", "wristcam",
    # control scheme
    "cent", "decent", "joint", "agent0", "agent1", "agent2", "arm0", "arm1", "arm2",
    # phase / job kind
    "train", "eval", "retrain", "resume", "calibration",
    # model
    "dp", "pi05",
)


def _derive_tags_from_name(name: str) -> list[str]:
    """Pick out structural tokens from a run name → list of wandb tags.

    Token matching is case-insensitive and word-boundary aware to avoid
    surprising substring hits ("dp" should not match "depth"; "agent0"
    should not match "agent00"). Result is deduplicated + sorted.
    """
    if not name:
        return []
    import re
    low = name.lower()
    out = set()
    for tok in _TAG_TOKENS:
        if re.search(rf"(?<![a-z0-9]){re.escape(tok)}(?![a-z0-9])", low):
            out.add(tok)
    return sorted(out)


class WandbRun:
    """Context manager around `wandb.init` / `wandb.finish`.

    `enabled=False` makes every method a no-op so callers can leave their
    existing `--wandb` flag wiring in place.
    """

    def __init__(
        self,
        *,
        enabled: bool,
        project: str,
        job_type: str,
        name: str,
        group: Optional[str] = None,
        tags: Optional[list] = None,
        config: Optional[dict] = None,
    ):
        self.enabled = enabled
        self.project = project
        self.job_type = job_type
        self.name = name
        self.group = group
        # Workflow improvement #3: auto-derive structural tags from the run
        # name so future filterability ("show me all decent TSC runs") is free
        # from day one and doesn't require retroactive `wandb api set-tags`
        # passes (the Tier-3 purge needed 9 such by-hand updates).
        derived = _derive_tags_from_name(name)
        explicit = list(tags) if tags else []
        self.tags = sorted(set(derived) | set(explicit))
        self.config = dict(config) if config else {}
        self.run = None

    def __enter__(self) -> "WandbRun":
        if not self.enabled:
            return self
        assert_wandb_live()
        import wandb  # local import keeps eval_context importable without wandb
        self.run = wandb.init(
            project=self.project,
            job_type=self.job_type,
            name=self.name,
            group=self.group,
            tags=self.tags,
            config=self.config,
        )
        if self.run is None:
            raise RuntimeError(
                "wandb.init() returned None — likely an authentication or "
                "network failure that wandb swallowed silently. Aborting."
            )
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if self.run is not None:
            import wandb
            wandb.finish(exit_code=0 if exc_type is None else 1)
            self.run = None
        return False  # never suppress exceptions

    def log_episode(self, idx: int, **fields) -> None:
        if self.run is None:
            return
        import wandb
        payload = {f"episode/{k}": v for k, v in fields.items()}
        payload["episode/idx"] = idx
        wandb.log(payload)

    def log_summary(self, **fields) -> None:
        if self.run is None:
            return
        import wandb
        wandb.log({f"summary/{k}": v for k, v in fields.items()})

    def log_raw(self, payload: dict) -> None:
        """Escape hatch for callers that want full control over the namespace."""
        if self.run is None:
            return
        import wandb
        wandb.log(payload)


class VideoRecorder:
    """Limit per-eval video output to the first N seeds.

    Plan v2 had every eval saving 60 videos (1/seed) — at ~50 MB each that's
    3 GB per eval and dozens of evals per week. Default is to record only the
    first 3 seeds; pass `all_seeds=True` for full coverage.

    Usage:
        with VideoRecorder(record_dir="/path", max_recorded=3) as videos:
            for i, seed in enumerate(seeds):
                vp = videos.video_path_for(i, seed)
                metrics = run_episode(..., video_path=vp)
    """

    def __init__(
        self,
        record_dir: str,
        max_recorded: int = 3,
        all_seeds: bool = False,
    ):
        self.record_dir = record_dir
        self.max_recorded = max_recorded
        self.all_seeds = all_seeds

    def __enter__(self) -> "VideoRecorder":
        if self.record_dir:
            os.makedirs(self.record_dir, exist_ok=True)
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    def video_path_for(
        self,
        seed_idx: int,
        seed_int: int,
        suffix: str = "",
        ext: str = ".mp4",
    ) -> Optional[str]:
        """Return a path to write to, or None when this seed exceeds the cap."""
        if not self.record_dir:
            return None
        if not self.all_seeds and seed_idx >= self.max_recorded:
            return None
        return os.path.join(
            self.record_dir, f"seed{seed_int:07d}{suffix}{ext}"
        )
