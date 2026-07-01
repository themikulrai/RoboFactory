#!/usr/bin/env python3
"""Render a field-notes "episode viewer" cell for subtask-conditioned
multi-arm episodes (LiftBarrier = 2 arms, ThreeRobotsStackCube = 3 arms),
with the PER-ARM SUBTASK overlaid on every frame.

Reads stored RGB arrays + the subtask stream from disk (NO GPU/SAPIEN),
tiles the cameras per frame, overlays a banner (variant/seed/frame + each
arm's live subtask, WAIT highlighted in orange/bold), encodes a small H.264
mp4 per episode, base64-embeds them, and POSTs an agent cell with a sandbox
visual to the "Data Aug" project via the field-notes HTTP API (large-payload
fallback).

Select the task with --task {LiftBarrier,ThreeRobotsStackCube}. Per-task
config (data dir, cameras, arms, layout, featured episodes, badge colors,
title/conclusion) lives in TASKS below.

Mirrors viewer cell 37206884 (LiftBarrier episode viewer) in the same project.
"""
from __future__ import annotations

import argparse
import base64
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import h5py
import numpy as np

# ---------------------------------------------------------------------------
# Shared render constants
# ---------------------------------------------------------------------------
TILE = 240  # common tile size (px)
FPS = 20
OUT_W = 760  # final scaled width
CRF = "30"   # libx264 quality knob (higher = smaller file)

# field-notes API (copied from render_eval_cell.py)
FN_BASE = "https://field-notes-mikulrai-5633eb0f870d.herokuapp.com"
FN_KEY = "78fc1af37b213bff1c9bc2c9c3b3897dc8ad4b7e2ff4cd110ef9f0e87e9c7428"
PROJECT_ID = "457a68e8-982b-476e-9918-30f8095cddf7"  # "Data Aug"

# colors (BGR for cv2)
WHITE = (255, 255, 255)
GREY = (180, 180, 180)
WAIT_COLOR = (40, 110, 240)  # orange/red-ish for WAIT
ARM_COLOR = (210, 235, 255)  # warm white for active subtask

# ---------------------------------------------------------------------------
# Per-task config
# ---------------------------------------------------------------------------
TASKS = {
    "LiftBarrier": {
        "data_dir": "/iris/u/mikulrai/datasets/multi_robot/RoboFactory/subtask_gen_validate5/LiftBarrier",
        "num_arms": 2,
        "featured": [0, 1, 3, 4, 5, 9],
        # (h5 cam key, tile label)
        "cams": [
            ("head_camera_global", "GLOBAL"),
            ("head_camera_agent0", "ag0"),
            ("head_camera_agent1", "ag1"),
            ("hand_camera_0", "hand0"),
            ("hand_camera_1", "hand1"),
        ],
        "global_label": "GLOBAL",
        "row_labels": ["ag0", "ag1", "hand0", "hand1"],
        "arm_prefixes": ["L (arm0): ", "R (arm1): "],
        "frame_stride": 1,
        "title": "LiftBarrier — Contrastive Pairs episode viewer (subtask overlay)",
        "conclusion": ("Pick an episode; the banner shows each arm's live subtask "
                       "— watch the follower 'wait' in stagger vs 'approach' in "
                       "simultaneous, same scene."),
        "cams_blurb": "cameras: GLOBAL, ag0, ag1, hand0, hand1 · overlay: per-arm subtask",
        "badge_colors": {"simultaneous": "#1f78b4", "stagger_a_leads": "#e6700d",
                         "stagger_b_leads": "#7a3fb0"},
        "after_cell_id": "fa30fe5b-8d77-4a43-9164-905e47e14fb3",
    },
    "ThreeRobotsStackCube": {
        "data_dir": "/iris/u/mikulrai/datasets/multi_robot/RoboFactory/subtask_gen_tsc_v4/ThreeRobotsStackCube",
        "num_arms": 3,
        # 3 DISTINCT behaviours only (traj_2 was raise_and_wait = byte-identical
        # duplicate of traj_0 simultaneous_pick, so it is DROPPED):
        #   traj_0 simultaneous_pick, traj_1 staggered_pick, traj_3 direct_place.
        "featured": [0, 1, 3],
        "cams": [
            ("head_camera_global", "GLOBAL"),
            ("head_camera_agent0", "ag0"),
            ("head_camera_agent1", "ag1"),
            ("head_camera_agent2", "ag2"),
            ("hand_camera_0", "hand0"),
            ("hand_camera_1", "hand1"),
            ("hand_camera_2", "hand2"),
        ],
        "global_label": "GLOBAL",
        # GLOBAL on top, then ag0/ag1/ag2 row, then hand0/hand1/hand2 row.
        "row_labels": ["ag0", "ag1", "ag2", "hand0", "hand1", "hand2"],
        "arm_prefixes": ["arm0/L (blue side): ", "arm1/R: ", "arm2/M (middle): "],
        "frame_stride": 4,  # T~450-530 → ~5-7s @20fps
        "out_w": 640,       # 7 tiles → smaller width keeps b64 < ~200KB
        "crf": "36",
        "title": "ThreeRobotsStackCube — Contrastive Pairs episode viewer (subtask overlay)",
        "conclusion": ("Three DISTINCT collision-free serial-stack behaviours "
                       "(blue->green->red), same seed. simultaneous_pick vs "
                       "staggered_pick differ in pick timing (followers 'wait' at "
                       "frame 0 in stagger); direct_place vs simultaneous_pick differ "
                       "in arm0's first placement (descend directly vs raise-then-"
                       "place). Pick an episode; banner shows each arm's live subtask."),
        "cams_blurb": ("cameras: GLOBAL, ag0/ag1/ag2 heads, hand0/hand1/hand2 "
                       "· overlay: per-arm subtask"),
        "badge_colors": {"simultaneous_pick": "#1f78b4", "staggered_pick": "#e6700d",
                         "direct_place": "#7a3fb0"},
        "after_cell_id": "fb91309c-04c3-4933-a969-7bab2fd08fb2",
    },
}


def log(m: str) -> None:
    print(f"[render_subtask_viewer] {m}", file=sys.stderr, flush=True)


def _dec(x) -> str:
    return x.decode("utf-8") if isinstance(x, (bytes, bytearray)) else str(x)


# ---------------------------------------------------------------------------
# Frame tiling + banner
# ---------------------------------------------------------------------------
def label_tile(img: np.ndarray, name: str) -> np.ndarray:
    """Resize a single camera frame to TILE x TILE and draw its cam name."""
    t = cv2.resize(img, (TILE, TILE), interpolation=cv2.INTER_AREA)
    t = cv2.cvtColor(t, cv2.COLOR_RGB2BGR)  # h5 is RGB; cv2 draws/writes BGR
    cv2.rectangle(t, (0, 0), (TILE - 1, TILE - 1), (60, 60, 60), 1)
    (tw, th), _ = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.rectangle(t, (4, 4), (8 + tw, 10 + th), (0, 0, 0), -1)
    cv2.putText(t, name, (6, 8 + th), cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 1,
                cv2.LINE_AA)
    return t


def compose_frame(tiles: dict[str, np.ndarray], cfg: dict) -> np.ndarray:
    """GLOBAL large/centered on top; remaining cams tiled in rows below.

    Heads and hands are split into separate rows when both present (TSC),
    else a single row (LB). The big GLOBAL tile spans the full row width.
    """
    g = tiles[cfg["global_label"]]
    row_labels = cfg["row_labels"]
    heads = [k for k in row_labels if k.startswith("ag")]
    hands = [k for k in row_labels if k.startswith("hand")]

    if heads and hands and len(heads) >= 2:
        rows = [heads, hands]
    else:
        rows = [row_labels]

    full_w = TILE * max(len(r) for r in rows)
    body_rows = []
    for r in rows:
        strip = np.hstack([tiles[k] for k in r])  # (TILE, len*TILE)
        if strip.shape[1] < full_w:  # left-pad/center shorter rows
            pad = full_w - strip.shape[1]
            x0 = pad // 2
            canvas = np.zeros((TILE, full_w, 3), dtype=np.uint8)
            canvas[:, x0:x0 + strip.shape[1]] = strip
            strip = canvas
        body_rows.append(strip)

    # big global tile: 2x, centered on the top band
    big = cv2.resize(g, (TILE * 2, TILE * 2), interpolation=cv2.INTER_AREA)
    top = np.zeros((TILE * 2, full_w, 3), dtype=np.uint8)
    x0 = (full_w - TILE * 2) // 2
    top[:, x0:x0 + TILE * 2] = big
    return np.vstack([top] + body_rows)


def draw_banner(frame: np.ndarray, l1: str, arm_texts: list[str],
                prefixes: list[str]) -> np.ndarray:
    """Banner below the frame: l1 then one line per arm. WAIT in WAIT_COLOR."""
    w = frame.shape[1]
    n = len(arm_texts)
    line_h = 30
    bh = 30 + n * line_h + 6
    banner = np.full((bh, w, 3), 22, dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(banner, l1, (10, 24), font, 0.6, WHITE, 1, cv2.LINE_AA)

    def line(prefix: str, text: str, y: int) -> None:
        is_wait = text.strip().lower() == "wait"
        col = WAIT_COLOR if is_wait else ARM_COLOR
        cv2.putText(banner, prefix, (10, y), font, 0.6, GREY, 1, cv2.LINE_AA)
        (pw, _), _ = cv2.getTextSize(prefix, font, 0.6, 1)
        disp = text.upper() if is_wait else text
        cv2.putText(banner, disp, (14 + pw, y), font, 0.6, col,
                    2 if is_wait else 1, cv2.LINE_AA)

    y = 30 + 24
    for i in range(n):
        line(prefixes[i], arm_texts[i], y)
        y += line_h
    return np.vstack([frame, banner])


# ---------------------------------------------------------------------------
# Per-episode render
# ---------------------------------------------------------------------------
def render_episode(rgb_f, stream_f, tid: int, variant: str, seed: int,
                   cfg: dict) -> str:
    g = rgb_f[f"traj_{tid}"]
    sg = stream_f[f"traj_{tid}"]
    n = cfg["num_arms"]
    arm_streams = [[_dec(x) for x in sg[f"subtask_arm{a}_text"][:]]
                   for a in range(n)]
    T = len(arm_streams[0])  # == actions length == rgb frames - 1
    stride = cfg["frame_stride"]
    prefixes = cfg["arm_prefixes"]

    rgbs = {label: g[f"obs/sensor_data/{key}/rgb"] for key, label in cfg["cams"]}

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        out_i = 0
        for t in range(0, T, stride):  # first T rgb frames align to subtask[t]
            tiles = {label: label_tile(rgbs[label][t], label)
                     for _, label in cfg["cams"]}
            frame = compose_frame(tiles, cfg)
            l1 = f"{variant}  |  seed {seed}  |  frame {t + 1}/{T}"
            frame = draw_banner(frame, l1, [arm_streams[a][t] for a in range(n)],
                                prefixes)
            cv2.imwrite(str(td / f"{out_i:04d}.png"), frame)
            out_i += 1

        out = td / "out.mp4"
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-framerate", str(FPS), "-i", str(td / "%04d.png"),
            "-vf", f"scale={cfg.get('out_w', OUT_W)}:-2",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast",
            "-crf", cfg.get("crf", CRF), "-movflags", "+faststart", "-an", str(out),
        ]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if r.returncode != 0 or not out.exists():
            raise RuntimeError(f"ffmpeg failed traj_{tid}: {r.stderr[:300]}")
        b64 = base64.b64encode(out.read_bytes()).decode("ascii")
    log(f"traj_{tid} {variant} seed{seed}: T={T} frames={out_i} "
        f"b64={len(b64)/1024:.1f}KB")
    return b64


# ---------------------------------------------------------------------------
# Cell builder + HTTP
# ---------------------------------------------------------------------------
def build_cell(eps: list[dict], cfg: dict) -> dict:
    html = (
        '<div class="wrap"><div class="head">'
        '<label class="pick">Episode<select id="ep"></select></label>'
        '<span id="badge" class="badge"></span>'
        f'<span class="cams">{cfg["cams_blurb"]}</span>'
        '<span class="ctrl"><button id="replay">replay</button></span>'
        '</div><video id="vid" controls autoplay loop muted playsinline></video>'
        '</div>'
    )
    badge_css = "".join(
        f".badge.{k}{{background:{v}}}" for k, v in cfg["badge_colors"].items())
    css = (
        "body{margin:0;font:13px/1.4 ui-sans-serif,system-ui,sans-serif;color:#111;"
        "background:#fafafa;padding:10px}"
        ".wrap{display:flex;flex-direction:column;gap:8px}"
        ".head{display:flex;align-items:center;gap:14px;flex-wrap:wrap;background:#fff;"
        "border:1px solid #e3e3e3;border-radius:6px;padding:8px 10px}"
        "label.pick{font-weight:600;font-size:12px}"
        "label.pick select{margin-left:6px;font:inherit;padding:3px 6px;border:1px solid #bbb;"
        "border-radius:4px;background:#fff;font-family:ui-monospace,Menlo,monospace;"
        "font-size:11px;min-width:320px}"
        ".badge{font-size:11px;font-weight:700;padding:2px 9px;border-radius:3px;color:#fff}"
        + badge_css +
        ".cams{font-family:ui-monospace,Menlo,monospace;font-size:11px;color:#666}"
        ".ctrl{margin-left:auto}"
        ".ctrl button{font:inherit;padding:3px 10px;border:1px solid #bbb;border-radius:4px;"
        "background:#fff;cursor:pointer}.ctrl button:hover{background:#eee}"
        "video{width:100%;height:auto;background:#000;border-radius:6px;border:1px solid #ddd}"
    )
    js = (
        "const EPISODES=" + json.dumps(eps, separators=(",", ":")) + ";\n" + r"""
const sel=document.getElementById('ep');
const vid=document.getElementById('vid');
const badge=document.getElementById('badge');
for(const e of EPISODES){
  const o=document.createElement('option');
  o.value=e.idx;
  o.textContent=`ep ${e.idx}  ·  ${e.kind.toUpperCase()}  ·  seed ${e.seed}`;
  sel.appendChild(o);
}
function load(){
  const e=EPISODES.find(x=>x.idx==parseInt(sel.value));
  if(!e)return;
  vid.src=e.video; vid.load(); vid.play().catch(()=>{});
  badge.textContent=e.kind;
  badge.className='badge '+e.kind;
}
sel.addEventListener('change',load);
document.getElementById('replay').onclick=()=>{vid.currentTime=0;vid.play();};
load();
""".strip()
    )
    return {
        "title": cfg["title"],
        "kind": "agent",
        "status": "open",
        "conclusion": cfg["conclusion"],
        "visual": {"kind": "sandbox", "html": html, "js": js, "css": css},
        "after_cell_id": cfg["after_cell_id"],
    }


def post_cell(cell: dict) -> dict:
    import urllib.request
    data = json.dumps(cell).encode()
    req = urllib.request.Request(
        f"{FN_BASE}/projects/{PROJECT_ID}/cells", data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("X-Field-Notes-Key", FN_KEY)
    with urllib.request.urlopen(req, timeout=180) as resp:
        return json.loads(resp.read().decode())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="ThreeRobotsStackCube", choices=list(TASKS))
    ap.add_argument("--dry-run", action="store_true",
                    help="render + report sizes but do NOT POST the cell")
    args = ap.parse_args()
    cfg = TASKS[args.task]

    data_dir = Path(cfg["data_dir"])
    rgb_h5 = data_dir / f"{args.task}.h5"
    stream_h5 = data_dir / f"{args.task}_subtask_stream.h5"
    meta_json = data_dir / "subtask_meta.json"

    meta = json.load(open(meta_json))
    by_id = {e["episode_id"]: e for e in meta["episodes"]}

    eps = []
    sizes = {}
    with h5py.File(rgb_h5, "r") as rf, h5py.File(stream_h5, "r") as sf:
        for tid in cfg["featured"]:
            if tid not in by_id:
                log(f"skip traj_{tid}: not in meta")
                continue
            m = by_id[tid]
            variant, seed = m["variant"], m["env_seed"]
            b64 = render_episode(rf, sf, tid, variant, seed, cfg)
            sizes[tid] = len(b64)
            eps.append({
                "idx": tid, "kind": variant, "seed": seed,
                "video": "data:video/mp4;base64," + b64,
            })

    cell = build_cell(eps, cfg)
    payload_bytes = len(json.dumps(cell).encode())
    log(f"total payload {payload_bytes/1024:.1f} KB")
    cid = None
    if not args.dry_run:
        log("POSTing...")
        out = post_cell(cell)
        cid = out.get("id")
        log(f"created cell id={cid}")
    else:
        log("dry-run: not posting")
    print(json.dumps({
        "task": args.task,
        "cell_id": cid,
        "sizes_b64": sizes,
        "payload_bytes": payload_bytes,
        "n_episodes": len(eps),
    }))
    return 0


if __name__ == "__main__":
    sys.exit(main())
