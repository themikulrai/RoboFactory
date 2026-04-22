"""Parallel orchestrator: renders all videos, computes stats, builds site.

Runs locally on iris-ws-17 (32 cores). Pure CPU pipeline: no GPU, no Slurm.
"""
from __future__ import annotations

import argparse
import json
import time
from multiprocessing import Pool
from pathlib import Path

from tools.viz.build_manifest import main as build_manifest_main
from tools.viz.build_site import build as build_site_main
from tools.viz.compute_stats import compute_task_stats
from tools.viz.render_videos import (
    DATA_ROOT, SITE_ROOT, discover_tasks, render_one, write_episodes_json,
)


def _render_worker(args):
    task, ep = args
    return render_one(task, ep)


def render_all(tasks: list[str], workers: int) -> None:
    h5_paths = {t: DATA_ROOT / t / f"{t}.h5" for t in tasks}
    import h5py
    work: list[tuple[str, int]] = []
    per_task: dict[str, list[tuple[str, int]]] = {t: [] for t in tasks}
    for t in tasks:
        with h5py.File(h5_paths[t], "r") as f:
            eps = sorted(int(k.split("_")[1]) for k in f.keys() if k.startswith("traj_"))
        for ep in eps:
            work.append((t, ep))
            per_task[t].append((t, ep))
    print(f"rendering {len(work)} episodes across {len(tasks)} tasks with {workers} workers")

    t0 = time.time()
    results_by_task: dict[str, list[dict]] = {t: [] for t in tasks}
    done = 0
    with Pool(workers) as pool:
        for row in pool.imap_unordered(_render_worker, work, chunksize=4):
            results_by_task[row["task"]].append(row)
            done += 1
            if done % 50 == 0:
                dt = time.time() - t0
                print(f"  {done}/{len(work)}  ({done/dt:.1f} eps/s)")
    for t, rows in results_by_task.items():
        write_episodes_json(t, rows)
    dt = time.time() - t0
    errs = sum(1 for rs in results_by_task.values() for r in rs if "error" in r)
    print(f"render done in {dt:.1f}s  ({len(work)/dt:.1f} eps/s)  errors={errs}")


def stats_all(tasks: list[str], workers: int) -> None:
    t0 = time.time()
    out_dir = SITE_ROOT / "stats"
    out_dir.mkdir(parents=True, exist_ok=True)
    with Pool(min(workers, len(tasks))) as pool:
        for stats in pool.imap_unordered(compute_task_stats, tasks):
            (out_dir / f"{stats['task']}.json").write_text(json.dumps(stats))
            print(f"  stats: {stats['task']} succ={stats['success_rate']:.2%}")
    print(f"stats done in {time.time()-t0:.1f}s")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default=None, help="comma-separated task names; default = all discovered")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--skip-render", action="store_true")
    ap.add_argument("--skip-stats", action="store_true")
    args = ap.parse_args()

    tasks = args.tasks.split(",") if args.tasks else discover_tasks()
    print(f"tasks: {tasks}")

    if not args.skip_render:
        render_all(tasks, args.workers)
    if not args.skip_stats:
        stats_all(tasks, args.workers)

    print("building manifest + site...")
    build_manifest_main()
    build_site_main()
    print(f"\nopen: {SITE_ROOT / 'index.html'}")


if __name__ == "__main__":
    main()
