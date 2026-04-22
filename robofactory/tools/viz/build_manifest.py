"""Collect per-episode metadata from render artifacts and per-task stats into a
single manifest.json that drives the single-page app."""
from __future__ import annotations

import json
import time
from pathlib import Path

SITE_ROOT = Path("/iris/u/mikulrai/data/RoboFactory/site")


def build() -> dict:
    media_root = SITE_ROOT / "media"
    stats_root = SITE_ROOT / "stats"
    tasks: list[dict] = []
    episodes: list[dict] = []
    for task_dir in sorted(d for d in media_root.iterdir() if d.is_dir()):
        task = task_dir.name
        stats_path = stats_root / f"{task}.json"
        if not stats_path.exists():
            print(f"[skip] no stats for {task}")
            continue
        stats = json.loads(stats_path.read_text())
        tasks.append({
            "name": task,
            "n_agents": stats["n_agents"],
            "n_episodes": stats["n_episodes"],
            "success_rate": stats["success_rate"],
            "length_mean": stats["length_mean"],
            "jerk_mean": stats["jerk_mean"],
            "diversity_l2": stats["diversity_l2"],
        })
        ep_meta_path = task_dir / "episodes.json"
        if not ep_meta_path.exists():
            print(f"[skip] no episodes.json for {task}")
            continue
        ep_rows = json.loads(ep_meta_path.read_text())
        for r in ep_rows:
            if "error" in r:
                continue
            episodes.append({
                "task": task,
                "ep": r["ep"],
                "success": r["success"],
                "length": r["length"],
                "jerk": r["mean_jerk"],
                "n_cams": r["n_cams"],
                "video": r["video"],
                "thumb": r["thumb"],
            })
    return {"tasks": tasks, "episodes": episodes, "build_id": int(time.time())}


def main() -> None:
    manifest = build()
    (SITE_ROOT / "manifest.json").write_text(json.dumps(manifest, separators=(",", ":")))
    print(f"manifest: {len(manifest['tasks'])} tasks, {len(manifest['episodes'])} episodes")


if __name__ == "__main__":
    main()
