"""Build a field-notes cell summarising a DART sigma sweep.

Reuses the H.264/yuv420p +faststart base64 transcode recipe from
robofactory/scripts/render_eval_cell.py (libx264, pix_fmt yuv420p, crf 30,
scale ~320px wide, +faststart, base64 data URI, ~8MB total cap) and the same
direct-HTTP field-notes I/O (X-Field-Notes-Key header), so it works from a
compute node without MCP.

Importable builders (the load-bearing parts):
  - ``transcode_b64(src)``            -> base64 H.264 mp4 string (or None)
  - ``keep_rate_png_b64(points)``     -> base64 PNG of keep-rate vs sigma
  - ``gather_sample_clips(video_dir, n)`` -> list of {name, b64} sample mp4s
  - ``load_keep_rates(meta_paths)``   -> [{sigma, kept, attempted, keep_rate}]

The cell-push (``post_cell``) is a thin wrapper around the same HTTP endpoints
render_eval_cell.py uses; if field-notes details drift, the builders above are
still correct and reusable on their own.

Usage:
    python scripts/dart/render_dart_cell.py \\
        --dart-meta .../sigma0.0/Task/dart_meta.json \\
                    .../sigma0.15/Task/dart_meta.json ... \\
        --video-dir .../dart_videos
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

# field-notes API (same Heroku app + key as render_eval_cell.py)
FN_BASE = "https://field-notes-mikulrai-5633eb0f870d.herokuapp.com"
PROJECT_ID = "457a68e8-982b-476e-9918-30f8095cddf7"
FN_KEY = "78fc1af37b213bff1c9bc2c9c3b3897dc8ad4b7e2ff4cd110ef9f0e87e9c7428"

MAX_GALLERY_B64_BYTES = 8_000_000
MAX_SAMPLE_CLIPS = 6


def log(msg: str) -> None:
    print(f"[render_dart_cell] {msg}", file=sys.stderr, flush=True)


# ----------------------------------------------------------------------------
# Video transcoding + base64 (copied recipe from render_eval_cell.transcode_b64)
# ----------------------------------------------------------------------------
def transcode_b64(src: Path) -> str | None:
    """Transcode an MP4 to browser-safe H.264/yuv420p +faststart, return base64
    string. Returns None on failure. Identical recipe to render_eval_cell.py."""
    src = Path(src)
    try:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tf:
            dst = Path(tf.name)
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
            "-vf", "scale=320:-2",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast",
            "-crf", "30", "-movflags", "+faststart", "-an", str(dst),
        ]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        if r.returncode != 0 or not dst.exists() or dst.stat().st_size == 0:
            log(f"ffmpeg failed for {src.name}: {r.stderr[:200]}")
            dst.unlink(missing_ok=True)
            return None
        data = dst.read_bytes()
        dst.unlink(missing_ok=True)
        return base64.b64encode(data).decode("ascii")
    except Exception as e:
        log(f"transcode error {src.name}: {e}")
        return None


# ----------------------------------------------------------------------------
# Keep-rate data + plot
# ----------------------------------------------------------------------------
def load_keep_rates(meta_paths: list[Path]) -> list[dict]:
    """Read dart_meta.json sidecars and compute keep-rate vs sigma.

    keep_rate = kept / attempted. ``kept`` = len(meta['episodes']) (only
    successes are flushed). ``attempted`` is the requested --num if recorded in
    a sibling, else falls back to kept (so keep_rate=1.0 when unknown).

    The runner does not currently persist 'attempted' in dart_meta.json, so we
    look for an optional 'attempted'/'num' key; if absent, keep_rate is
    reported as kept/kept=1.0 and a note is set. (Document this clearly.)
    """
    points = []
    for mp in meta_paths:
        mp = Path(mp)
        try:
            meta = json.loads(mp.read_text())
        except Exception as e:
            log(f"could not read {mp}: {e}")
            continue
        kept = len(meta.get("episodes", []))
        # keep-rate is kept / requested-episodes (num). 'attempted' (total
        # solver runs incl. retries) is a different denominator; prefer 'num'
        # when present so keep-rate is per-requested-seed.
        attempted = int(meta.get("num", meta.get("attempted", kept)) or kept)
        attempted = max(attempted, kept)  # guard
        keep_rate = (kept / attempted) if attempted else 0.0
        points.append({
            "sigma": float(meta.get("sigma", 0.0)),
            "kept": kept,
            "attempted": attempted,
            "keep_rate": round(keep_rate, 4),
            "meta_path": str(mp),
        })
    points.sort(key=lambda p: p["sigma"])
    return points


def keep_rate_png_b64(points: list[dict]) -> str | None:
    """Matplotlib PNG (base64) of keep-rate vs sigma. Returns None on failure."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        log(f"matplotlib unavailable: {e}")
        return None
    if not points:
        return None
    xs = [p["sigma"] for p in points]
    ys = [100.0 * p["keep_rate"] for p in points]
    fig, ax = plt.subplots(figsize=(5, 3.2), dpi=120)
    ax.plot(xs, ys, "-o", color="#1f78b4", lw=2)
    for p in points:
        ax.annotate(f"{p['kept']}/{p['attempted']}",
                    (p["sigma"], 100.0 * p["keep_rate"]),
                    textcoords="offset points", xytext=(0, 6),
                    ha="center", fontsize=8, color="#555")
    ax.set_xlabel("disturbance sigma (rad std-dev)")
    ax.set_ylabel("keep-rate (success) %")
    ax.set_title("DART keep-rate vs disturbance magnitude")
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def gather_sample_clips(video_dir: Path, n: int = MAX_SAMPLE_CLIPS) -> list[dict]:
    """Transcode up to n sample recovery mp4s under video_dir to base64,
    respecting the total payload cap. Returns [{name, b64}]."""
    video_dir = Path(video_dir)
    if not video_dir.exists():
        return []
    mp4s = sorted(video_dir.rglob("*.mp4"))
    if len(mp4s) > n:
        step = len(mp4s) / n
        mp4s = [mp4s[int(i * step)] for i in range(n)]
    clips, total = [], 0
    for src in mp4s:
        b64 = transcode_b64(src)
        if b64 is None:
            continue
        if total + len(b64) > MAX_GALLERY_B64_BYTES:
            break
        total += len(b64)
        clips.append({"name": src.name, "b64": b64})
    return clips


# ----------------------------------------------------------------------------
# Cell content + field-notes HTTP I/O (mirrors render_eval_cell.py)
# ----------------------------------------------------------------------------
def build_cell(points: list[dict], png_b64: str | None,
               clips: list[dict]) -> dict:
    total_kept = sum(p["kept"] for p in points)
    total_att = sum(p["attempted"] for p in points)
    rows = "".join(
        f"<tr><td>{p['sigma']:.2f}</td><td>{p['kept']}</td>"
        f"<td>{p['attempted']}</td><td>{100.0*p['keep_rate']:.1f}%</td></tr>"
        for p in points
    )
    img_html = (
        f'<img style="max-width:520px;width:100%" '
        f'src="data:image/png;base64,{png_b64}"/>'
        if png_b64 else '<div class="hint">keep-rate plot unavailable</div>'
    )
    vids_html = "".join(
        f'<div class="vid"><video controls preload="metadata" muted '
        f'src="data:video/mp4;base64,{c["b64"]}"></video>'
        f'<div class="cap">{c["name"]}</div></div>'
        for c in clips
    ) or '<div class="hint">no sample clips available</div>'

    html = f"""
<div class="wrap">
  <div class="hdr">DART OOD-recovery data-gen — kept {total_kept}/{total_att}
    across {len(points)} sigma values</div>
  <div class="card"><div class="ct">Keep-rate vs disturbance sigma</div>
    {img_html}
    <table><thead><tr><th>sigma</th><th>kept</th><th>attempted</th>
      <th>keep-rate</th></tr></thead><tbody>{rows}</tbody></table>
  </div>
  <div class="card"><div class="ct">Sample recovery rollouts</div>
    <div class="gallery">{vids_html}</div>
  </div>
</div>
""".strip()

    css = """
.wrap{font-family:-apple-system,Segoe UI,Roboto,sans-serif;color:#1a1a1a;padding:4px}
.hdr{font-size:14px;font-weight:700;margin-bottom:8px}
.card{border:1px solid #e2e2e2;border-radius:8px;padding:8px;background:#fff;margin-bottom:10px}
.ct{font-size:12px;font-weight:700;margin-bottom:6px;color:#333}
table{width:100%;border-collapse:collapse;font-size:11px;margin-top:6px}
th,td{padding:3px 6px;text-align:center;border-bottom:1px solid #eee}
th{background:#f6f6f6}
.gallery{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:8px}
.vid{border:1px solid #ddd;border-radius:6px;overflow:hidden}
.vid video{width:100%;display:block;background:#000}
.vid .cap{font-size:10px;padding:3px 6px;font-weight:600}
.hint{font-size:11px;color:#999}
""".strip()

    metrics = [
        {"k": "DART kept", "v": f"{total_kept}/{total_att}",
         "d": f"across {len(points)} sigma values"},
    ]
    for p in points:
        metrics.append({
            "k": f"sigma {p['sigma']:.2f}",
            "v": f"{p['kept']}/{p['attempted']} ({100.0*p['keep_rate']:.0f}%)",
            "d": "kept / attempted",
        })

    return {
        "title": "DART OOD-recovery — keep-rate vs disturbance sigma",
        "kind": "agent",
        "status": "in_progress",
        "conclusion": (
            f"DART disturbance sweep: kept {total_kept}/{total_att} demos across "
            f"{len(points)} sigma values; larger sigma should lower keep-rate."
        ),
        "metrics": metrics,
        "visual": {"kind": "sandbox", "html": html, "js": "", "css": css},
    }


def _req(method: str, url: str, payload: dict | None) -> dict:
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Content-Type", "application/json")
    req.add_header("X-Field-Notes-Key", FN_KEY)
    with urllib.request.urlopen(req, timeout=120) as resp:
        body = resp.read().decode()
        return json.loads(body) if body else {}


def post_cell(cell: dict, after_cell_id: str | None = None) -> str:
    """Create a new field-notes cell in the DART project. Returns the cell id."""
    payload = dict(cell)
    if after_cell_id:
        payload["after_cell_id"] = after_cell_id
    out = _req("POST", f"{FN_BASE}/projects/{PROJECT_ID}/cells", payload)
    cid = out.get("id")
    if not cid:
        raise RuntimeError(f"create returned no id: {out}")
    return cid


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dart-meta", nargs="+", required=True, dest="dart_meta",
                    help="one or more dart_meta.json paths (one per sigma)")
    ap.add_argument("--video-dir", default=None, dest="video_dir",
                    help="dir of sample recovery mp4s to embed (optional)")
    ap.add_argument("--after-cell-id", default=None, dest="after_cell_id")
    ap.add_argument("--no-post", action="store_true",
                    help="build everything but do NOT push to field-notes")
    args = ap.parse_args()

    points = load_keep_rates([Path(p) for p in args.dart_meta])
    png_b64 = keep_rate_png_b64(points)
    clips = gather_sample_clips(Path(args.video_dir)) if args.video_dir else []
    cell = build_cell(points, png_b64, clips)

    if args.no_post:
        log(f"--no-post: built cell with {len(points)} points, "
            f"{len(clips)} clips, png={'yes' if png_b64 else 'no'}")
        print(json.dumps({"points": points, "n_clips": len(clips)}, indent=2))
        return 0

    cid = post_cell(cell, after_cell_id=args.after_cell_id)
    log(f"posted cell {cid}")
    print(cid)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        log(f"FATAL: {type(e).__name__}: {e}")
        sys.exit(0)
