"""Read-side checkpoint resolver.

Plan v2 C2 S4: a JSONL index `runs/ckpt_index.jsonl` maps
`(framework, dataset, task, identifier, epoch) → ckpt_path`. Scanner walks
both `/iris/u/mikulrai/checkpoints/{RoboFactory,openpi}/` once and writes
one row per `.ckpt` file (DP) or step subdir (Pi0.5). Existing 2 TB of
ckpts stay where they are; reads consult the JSONL.

Why a flat JSONL instead of CSV: ckpts have variable metadata (extra
arms for decent runs, optional config tags); JSONL accommodates that
without forcing a schema migration on every new field.

API:
    resolver = CkptResolver()
    resolver.scan_legacy_tree(out_path="/iris/u/mikulrai/runs/ckpt_index.jsonl")
    paths = resolver.find(framework="dp", task="PickMeat", epoch=300)
"""
from __future__ import annotations

import dataclasses
import json
import re
from pathlib import Path
from typing import Iterable, Iterator, Optional


CKPT_ROOTS = (
    Path("/iris/u/mikulrai/checkpoints/RoboFactory"),
    Path("/iris/u/mikulrai/checkpoints/openpi"),
)
DEFAULT_INDEX_PATH = Path("/iris/u/mikulrai/runs/ckpt_index.jsonl")


@dataclasses.dataclass(frozen=True)
class CkptEntry:
    framework: str  # "dp" or "pi05"
    task: str       # PickMeat, ThreeRobotsStackCube, ...
    identifier: str # full ckpt-dir name (e.g. "PickMeat-rf_150" or "...arm0")
    epoch: int      # for DP: int from the .ckpt filename; for Pi0.5: orbax step
    path: str       # absolute filesystem path
    size_bytes: int
    mtime_unix: float

    def to_jsonl(self) -> str:
        return json.dumps(dataclasses.asdict(self))


# Match DP-style numeric ckpt files: 5.ckpt, 300.ckpt, 300_in1k.ckpt, ...
_DP_NUMERIC_CKPT = re.compile(r"^(\d+)(?:_[a-zA-Z0-9_-]+)?\.ckpt$")
# Pi0.5 orbax step dirs are non-zero-padded ints
_PI05_STEP_DIR = re.compile(r"^\d+$")


def _scan_dp(root: Path) -> Iterator[CkptEntry]:
    if not root.is_dir():
        return
    for run_dir in root.iterdir():
        if not run_dir.is_dir():
            continue
        identifier = run_dir.name
        # task = leading "<task>-rf" segment if present, else first underscore segment
        task = identifier.split("-rf")[0] if "-rf" in identifier else identifier.split("_")[0]
        for sub in run_dir.rglob("*.ckpt"):
            if not sub.is_file():
                continue
            m = _DP_NUMERIC_CKPT.match(sub.name)
            if not m:
                continue
            try:
                epoch = int(m.group(1))
                stat = sub.stat()
            except (OSError, ValueError):
                continue
            yield CkptEntry(
                framework="dp",
                task=task,
                identifier=identifier,
                epoch=epoch,
                path=str(sub),
                size_bytes=stat.st_size,
                mtime_unix=stat.st_mtime,
            )


def _scan_pi05(root: Path) -> Iterator[CkptEntry]:
    if not root.is_dir():
        return
    # Pi0.5 layout: <root>/<config>/<exp_name>/<step>/{params,assets,train_state}
    for config_dir in root.iterdir():
        if not config_dir.is_dir():
            continue
        for exp_dir in config_dir.iterdir():
            if not exp_dir.is_dir() or exp_dir.name.startswith("."):
                continue
            identifier = f"{config_dir.name}/{exp_dir.name}"
            task = config_dir.name
            for step_dir in exp_dir.iterdir():
                if not step_dir.is_dir() or not _PI05_STEP_DIR.match(step_dir.name):
                    continue
                if step_dir.name.endswith(".trash") or "trash_" in step_dir.name:
                    continue
                try:
                    stat = step_dir.stat()
                except OSError:
                    continue
                # rough size = sum of file sizes inside (capped to avoid huge walks)
                size = 0
                for f in step_dir.rglob("*"):
                    try:
                        size += f.stat().st_size
                    except OSError:
                        continue
                yield CkptEntry(
                    framework="pi05",
                    task=task,
                    identifier=identifier,
                    epoch=int(step_dir.name),
                    path=str(step_dir),
                    size_bytes=size,
                    mtime_unix=stat.st_mtime,
                )


class CkptResolver:
    """In-memory index of checkpoints, optionally backed by `ckpt_index.jsonl`."""

    def __init__(self, entries: Optional[Iterable[CkptEntry]] = None):
        self._entries: list[CkptEntry] = list(entries) if entries else []

    @classmethod
    def load(cls, path: Path | str = DEFAULT_INDEX_PATH) -> "CkptResolver":
        path = Path(path)
        if not path.is_file():
            return cls()
        entries: list[CkptEntry] = []
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entries.append(CkptEntry(**json.loads(line)))
        return cls(entries)

    def scan_legacy_tree(
        self,
        roots: Iterable[Path] = CKPT_ROOTS,
        out_path: Path | str = DEFAULT_INDEX_PATH,
    ) -> int:
        """Walk roots, write one JSONL row per ckpt to `out_path`. Returns count."""
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        n = 0
        with out_path.open("w") as f:
            for root in roots:
                root = Path(root)
                if "openpi" in str(root):
                    iter_ = _scan_pi05(root)
                else:
                    iter_ = _scan_dp(root)
                for entry in iter_:
                    f.write(entry.to_jsonl() + "\n")
                    n += 1
        # also keep entries in memory
        self._entries = list(self._iter_loaded(out_path))
        return n

    @staticmethod
    def _iter_loaded(path: Path) -> Iterator[CkptEntry]:
        with path.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    yield CkptEntry(**json.loads(line))

    def find(
        self,
        *,
        framework: Optional[str] = None,
        task: Optional[str] = None,
        identifier_substr: Optional[str] = None,
        epoch: Optional[int] = None,
    ) -> list[CkptEntry]:
        out = []
        for e in self._entries:
            if framework and e.framework != framework:
                continue
            if task and e.task != task:
                continue
            if identifier_substr and identifier_substr not in e.identifier:
                continue
            if epoch is not None and e.epoch != epoch:
                continue
            out.append(e)
        return out

    def latest(
        self,
        *,
        framework: Optional[str] = None,
        task: Optional[str] = None,
        identifier_substr: Optional[str] = None,
    ) -> Optional[CkptEntry]:
        candidates = self.find(framework=framework, task=task, identifier_substr=identifier_substr)
        if not candidates:
            return None
        return max(candidates, key=lambda e: (e.epoch, e.mtime_unix))

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self) -> Iterator[CkptEntry]:
        return iter(self._entries)


def main() -> int:
    """CLI: scan and write the index. Run from anywhere."""
    import argparse
    ap = argparse.ArgumentParser(description="Scan checkpoint roots into a JSONL index.")
    ap.add_argument("--roots", nargs="*", default=[str(r) for r in CKPT_ROOTS])
    ap.add_argument("--out", default=str(DEFAULT_INDEX_PATH))
    args = ap.parse_args()
    r = CkptResolver()
    n = r.scan_legacy_tree(roots=[Path(p) for p in args.roots], out_path=args.out)
    print(f"Wrote {n} entries to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
