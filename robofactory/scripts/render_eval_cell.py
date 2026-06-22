#!/usr/bin/env python3
"""Idempotent renderer for the Pi0.5 2SC decent-wristcam proprio-dropout
step-8k 60-seed eval (slurm job 15529328).

Reads the CURRENT eval state from disk and creates-or-updates ONE field-notes
cell in the "Stacking Cubes" project. Safe to call repeatedly from a polling
loop (every ~60s): cheap when nothing changed, never duplicates the cell.

What it builds in the cell sandbox (Chart.js + vanilla JS):
  (a) cumulative success-rate curve vs seeds-completed
  (b) per-seed table (seed | success | env-steps | wall-time) with running header
  (c) curated video gallery: ALL successful seeds + a sample of ~6 failures,
      base64-embedded, transcoded to H.264/yuv420p +faststart via ffmpeg
  (d) interactive per-step action viewer: pick any seed, plot per-arm joint
      deltas + gripper over timesteps (downsampled)

This is a PRELIMINARY eval: checkpoint step 8,000 of 20,000 (~40% trained).
Baseline to beat: 5/60 (8.3%) for the no-dropout 2SC decent wristcam policy.

Usage (a polling loop calls this exact command):
    /iris/u/mikulrai/data/miniforge3/envs/RoboFactory/bin/python \
        /iris/u/mikulrai/projects/RoboFactory/robofactory/scripts/render_eval_cell.py

Exit code 0 always (so a loop is not killed by a transient hiccup); problems
are printed to stderr.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

# ----------------------------------------------------------------------------
# Config (all absolute; cwd is not relied upon)
# ----------------------------------------------------------------------------
JOB_ID = "15529328"
CKPT_STEP = 8000
CKPT_TOTAL = 20000
BASELINE_TXT = "5/60 (8.3%) — no-dropout 2SC decent wristcam"

LOG_DIR = Path("/iris/u/mikulrai/logs/eval_pi05_2sc_wc_decent")
# iris-mcp redirects job stdout to /iris/u/mikulrai/slurm/<jobid>.out regardless
# of the script's #SBATCH --output path. The per-episode [seed_base=...] lines
# land there; the %x_%j.err path (below) still gets stderr.
OUT_LOG = Path(f"/iris/u/mikulrai/slurm/{JOB_ID}.out")
OUT_LOG_ALT = LOG_DIR / f"eval-pi05-2sc-pd-s8000_{JOB_ID}.out"
ERR_LOG = LOG_DIR / f"eval-pi05-2sc-pd-s8000_{JOB_ID}.err"
VIDEO_DIR = LOG_DIR / f"videos_{JOB_ID}"
TRAJ_PATH = LOG_DIR / f"trajectory_{JOB_ID}.jsonl"
CELL_ID_FILE = LOG_DIR / f"cell_id_{JOB_ID}.txt"
CONTENT_HASH_FILE = LOG_DIR / f"render_hash_{JOB_ID}.txt"
TASK = "TwoRobotsStackCube-rf"
TOTAL_SEEDS = 60

# field-notes API
FN_BASE = "https://field-notes-mikulrai-5633eb0f870d.herokuapp.com"
PROJECT_ID = "3798978d-74c3-43c9-93f8-4bf93d7abb6b"
AFTER_CELL_ID = "f4689a82-7243-48a0-90e9-648cef99d15f"
FN_KEY = "78fc1af37b213bff1c9bc2c9c3b3897dc8ad4b7e2ff4cd110ef9f0e87e9c7428"

# video gallery limits
MAX_FAILURE_VIDEOS = 6
# hard cap on total base64 video bytes embedded in the cell (~8 MB of b64).
# successes are always kept; failures are dropped first if over budget.
MAX_GALLERY_B64_BYTES = 8_000_000
# trajectory downsampling: keep at most this many step-points per seed plot
TRAJ_MAX_POINTS = 120

# per-episode stdout line, e.g.:
#   [seed_base=123 ep=000] success=True steps=187 wall_s=64.2
EPISODE_RE = re.compile(
    r"\[seed_base=(\d+)\s+ep=(\d+)\]\s+success=(True|False)\s+"
    r"steps=(\d+)\s+wall_s=([-\d.]+)"
)


def log(msg: str) -> None:
    print(f"[render_eval_cell] {msg}", file=sys.stderr, flush=True)


# ----------------------------------------------------------------------------
# State collection
# ----------------------------------------------------------------------------
def read_final_json() -> list[dict] | None:
    """If the eval finished, the authoritative results JSON exists. Pick the
    newest one whose args.run_id matches this job (run_id is the slurm id)."""
    best = None
    for fp in LOG_DIR.glob(f"eval_decent_{TASK}_*.json"):
        try:
            d = json.loads(fp.read_text())
        except Exception:
            continue
        if str(d.get("args", {}).get("run_id", "")) != JOB_ID:
            continue
        if best is None or fp.stat().st_mtime > best[0]:
            best = (fp.stat().st_mtime, d.get("results", []))
    return best[1] if best else None


def parse_episode_log() -> list[dict]:
    """Parse per-episode results live from the .out / .err logs."""
    rows: dict[int, dict] = {}
    for fp in (OUT_LOG, OUT_LOG_ALT, ERR_LOG):
        if not fp.exists():
            continue
        try:
            text = fp.read_text(errors="replace")
        except Exception:
            continue
        for m in EPISODE_RE.finditer(text):
            seed = int(m.group(1))
            rows[seed] = {
                "seed": seed,
                "success": m.group(3) == "True",
                "steps": int(m.group(4)),
                "wall_s": float(m.group(5)),
            }
    return [rows[s] for s in sorted(rows)]


def collect_results() -> tuple[list[dict], bool]:
    """Return (results, finished). Prefer the final JSON; else parse the log."""
    final = read_final_json()
    if final is not None:
        norm = []
        for r in final:
            # final json sometimes stores seed inflated (seed_base*100000);
            # normalise to the 100-159 range when that happened.
            seed = int(r.get("seed", 0))
            if seed >= 100000:
                seed = seed // 100000
            norm.append(
                {
                    "seed": seed,
                    "success": bool(r.get("success", False)),
                    "steps": int(r.get("steps", 0)),
                    "wall_s": float(r.get("wall_s", -1)),
                }
            )
        norm.sort(key=lambda r: r["seed"])
        return norm, True
    return parse_episode_log(), False


def job_running() -> bool:
    try:
        out = subprocess.run(
            ["sacct", "-j", JOB_ID, "-n", "-o", "State", "-X"],
            capture_output=True, text=True, timeout=20,
        ).stdout
        return "RUNNING" in out or "PENDING" in out
    except Exception:
        return False  # sacct often absent on login node; treat as unknown/done


# ----------------------------------------------------------------------------
# Trajectory data (per-step actions, all 60 seeds, downsampled)
# ----------------------------------------------------------------------------
def load_trajectories() -> dict[int, dict]:
    """Parse trajectory_<job>.jsonl -> {seed: {steps:[...],
    arm0:[[8]...], arm1:[[8]...]}} using applied_action_per_arm.
    Downsampled to <= TRAJ_MAX_POINTS points per seed to keep payload small."""
    if not TRAJ_PATH.exists():
        return {}
    raw: dict[int, list[dict]] = {}
    try:
        with TRAJ_PATH.open(errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    continue  # last line may be partially written
                # env reset multiplies seed_base by 100000 — normalise back.
                seed = int(row["seed"])
                if seed >= 100000:
                    seed //= 100000
                raw.setdefault(seed, []).append(row)
    except Exception as e:
        log(f"trajectory read failed: {e}")
        return {}

    out: dict[int, dict] = {}
    for seed, rows in raw.items():
        rows.sort(key=lambda r: r.get("step", 0))
        n = len(rows)
        if n == 0:
            continue
        stride = max(1, n // TRAJ_MAX_POINTS)
        sel = rows[::stride]
        steps = [int(r.get("step", i)) for i, r in enumerate(sel)]
        a0, a1 = [], []
        for r in sel:
            ap = r.get("applied_action_per_arm") or [[], []]
            a0.append([round(float(v), 5) for v in (ap[0] if len(ap) > 0 else [])])
            a1.append([round(float(v), 5) for v in (ap[1] if len(ap) > 1 else [])])
        out[seed] = {"steps": steps, "arm0": a0, "arm1": a1, "n_steps": n}
    return out


# ----------------------------------------------------------------------------
# Video transcoding + base64 embedding
# ----------------------------------------------------------------------------
def video_path_for(seed: int) -> Path | None:
    p = VIDEO_DIR / f"{TASK}_run{JOB_ID}_seed{seed}_ep000.mp4"
    return p if p.exists() else None


def transcode_b64(src: Path) -> str | None:
    """Transcode an eval MP4 to browser-safe H.264/yuv420p +faststart,
    return base64 string. Returns None on failure."""
    try:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tf:
            dst = Path(tf.name)
        # scale to 320px wide (keep aspect, even dims) + crf 30 to keep the
        # base64 payload small — the cell embeds many clips inline.
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


def build_gallery(results: list[dict]) -> list[dict]:
    """Curate videos: all successes + up to MAX_FAILURE_VIDEOS failures
    (evenly sampled across the failure seeds)."""
    succ = [r["seed"] for r in results if r["success"]]
    fail = [r["seed"] for r in results if not r["success"]]
    if len(fail) > MAX_FAILURE_VIDEOS:
        step = len(fail) / MAX_FAILURE_VIDEOS
        fail = [fail[int(i * step)] for i in range(MAX_FAILURE_VIDEOS)]
    chosen = [(s, True) for s in succ] + [(s, False) for s in fail]

    gallery = []
    for seed, ok in chosen:
        src = video_path_for(seed)
        if src is None:
            continue
        b64 = transcode_b64(src)
        if b64 is None:
            continue
        gallery.append({"seed": seed, "success": ok, "b64": b64})

    # enforce total payload budget: drop failures first (keep all successes),
    # then trim oldest successes if still over.
    def total() -> int:
        return sum(len(x["b64"]) for x in gallery)

    while total() > MAX_GALLERY_B64_BYTES and any(not x["success"] for x in gallery):
        for i in range(len(gallery) - 1, -1, -1):
            if not gallery[i]["success"]:
                del gallery[i]
                break
    while total() > MAX_GALLERY_B64_BYTES and len(gallery) > 1:
        del gallery[-1]
    return gallery


# ----------------------------------------------------------------------------
# Cell content builder
# ----------------------------------------------------------------------------
def build_cell(results: list[dict], traj: dict, gallery: list[dict],
               finished: bool, running: bool) -> dict:
    n = len(results)
    n_succ = sum(1 for r in results if r["success"])
    sr = (100.0 * n_succ / n) if n else 0.0

    # cumulative SR curve
    cum_x, cum_y, run_succ = [], [], 0
    for i, r in enumerate(results, 1):
        run_succ += int(r["success"])
        cum_x.append(i)
        cum_y.append(round(100.0 * run_succ / i, 2))

    if finished:
        state_txt = f"COMPLETE — {n}/{TOTAL_SEEDS} seeds"
    elif running:
        state_txt = f"RUNNING — {n}/{TOTAL_SEEDS} seeds done"
    else:
        state_txt = f"{n}/{TOTAL_SEEDS} seeds done (job state unknown)"

    data = {
        "meta": {
            "job": JOB_ID,
            "ckpt_step": CKPT_STEP,
            "ckpt_total": CKPT_TOTAL,
            "n_done": n,
            "n_total": TOTAL_SEEDS,
            "n_succ": n_succ,
            "sr": round(sr, 2),
            "baseline": BASELINE_TXT,
            "state": state_txt,
            "finished": finished,
        },
        "results": results,
        "cum": {"x": cum_x, "y": cum_y},
        "traj": traj,
        "gallery": gallery,
    }

    html = """
<div class="wrap">
  <div class="banner">PRELIMINARY — intermediate checkpoint step 8,000 of 20,000 (~40% trained). Results are not final.</div>
  <div class="hdr" id="hdr">loading…</div>
  <div class="grid2">
    <div class="card">
      <div class="ct">Cumulative success rate vs seeds completed</div>
      <canvas id="srChart"></canvas>
    </div>
    <div class="card">
      <div class="ct">Per-seed results</div>
      <div class="tablebox"><table id="seedTable">
        <thead><tr><th>seed</th><th>success</th><th>env-steps</th><th>wall&nbsp;s</th></tr></thead>
        <tbody></tbody></table></div>
    </div>
  </div>
  <div class="card">
    <div class="ct">Rollout videos — all successes + sample of failures (<span id="galCount">0</span> clips)</div>
    <div id="gallery" class="gallery"></div>
  </div>
  <div class="card">
    <div class="ct">Per-step action viewer
      &nbsp;seed <select id="seedSel"></select>
      &nbsp;arm <select id="armSel"><option value="0">arm 0</option><option value="1">arm 1</option></select>
      <span class="hint">7 joint deltas + gripper; downsampled</span>
    </div>
    <canvas id="actChart"></canvas>
    <div id="trajNote" class="hint"></div>
  </div>
</div>
""".strip()

    css = """
.wrap{font-family:-apple-system,Segoe UI,Roboto,sans-serif;color:#1a1a1a;padding:4px}
.banner{background:#fff3cd;border:1px solid #ffe08a;color:#7a5b00;font-weight:600;
  padding:6px 10px;border-radius:6px;margin-bottom:8px;font-size:12px}
.hdr{font-size:14px;font-weight:700;margin-bottom:8px}
.hdr .ok{color:#1a7f37}.hdr .bl{color:#888;font-weight:500}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:10px}
.card{border:1px solid #e2e2e2;border-radius:8px;padding:8px;background:#fff;margin-bottom:10px}
.ct{font-size:12px;font-weight:700;margin-bottom:6px;color:#333}
canvas{max-height:240px}
.tablebox{max-height:240px;overflow:auto}
table{width:100%;border-collapse:collapse;font-size:11px}
th,td{padding:3px 6px;text-align:center;border-bottom:1px solid #eee}
th{position:sticky;top:0;background:#f6f6f6}
td.s-ok{color:#1a7f37;font-weight:700}td.s-no{color:#b3261e}
.gallery{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:8px}
.vid{border:1px solid #ddd;border-radius:6px;overflow:hidden}
.vid video{width:100%;display:block;background:#000}
.vid .cap{font-size:11px;padding:3px 6px;font-weight:600}
.vid .cap.ok{background:#e7f6ec;color:#1a7f37}
.vid .cap.no{background:#fde8e6;color:#b3261e}
.hint{font-size:10px;color:#999;font-weight:400;margin-left:6px}
select{font-size:11px}
""".strip()

    js = "const D=" + json.dumps(data, separators=(",", ":")) + ";\n" + r"""
(function(){
  const M=D.meta;
  const hdr=document.getElementById('hdr');
  hdr.innerHTML='<span class="ok">'+M.n_succ+'/'+M.n_done+' successes</span> '+
    '&nbsp;SR='+M.sr+'%&nbsp;('+M.state+')'+
    '&nbsp;<span class="bl">ckpt step '+M.ckpt_step+'/'+M.ckpt_total+
    ' · baseline '+M.baseline+'</span>';

  // (a) cumulative SR chart
  new Chart(document.getElementById('srChart'),{
    type:'line',
    data:{labels:D.cum.x,datasets:[
      {label:'cumulative SR %',data:D.cum.y,borderColor:'#1f78b4',
       backgroundColor:'rgba(31,120,180,.12)',fill:true,tension:.15,
       pointRadius:2},
      {label:'baseline 8.3%',data:D.cum.x.map(()=>8.3),borderColor:'#999',
       borderDash:[5,4],pointRadius:0,fill:false}
    ]},
    options:{responsive:true,maintainAspectRatio:false,
      scales:{y:{min:0,max:100,title:{display:true,text:'success rate %'}},
              x:{title:{display:true,text:'seeds completed'}}},
      plugins:{legend:{labels:{boxWidth:12,font:{size:10}}}}}
  });

  // (b) per-seed table
  const tb=document.querySelector('#seedTable tbody');
  D.results.forEach(r=>{
    const tr=document.createElement('tr');
    tr.innerHTML='<td>'+r.seed+'</td>'+
      '<td class="'+(r.success?'s-ok':'s-no')+'">'+(r.success?'✓':'✗')+'</td>'+
      '<td>'+r.steps+'</td>'+
      '<td>'+(r.wall_s>=0?r.wall_s.toFixed(1):'-')+'</td>';
    tb.appendChild(tr);
  });
  if(!D.results.length){
    const tr=document.createElement('tr');
    tr.innerHTML='<td colspan="4" class="hint">no seeds completed yet</td>';
    tb.appendChild(tr);
  }

  // (c) video gallery
  const g=document.getElementById('gallery');
  document.getElementById('galCount').textContent=D.gallery.length;
  D.gallery.forEach(v=>{
    const d=document.createElement('div');d.className='vid';
    d.innerHTML='<video controls preload="metadata" muted '+
      'src="data:video/mp4;base64,'+v.b64+'"></video>'+
      '<div class="cap '+(v.success?'ok':'no')+'">seed '+v.seed+' · '+
      (v.success?'SUCCESS':'fail')+'</div>';
    g.appendChild(d);
  });
  if(!D.gallery.length){
    g.innerHTML='<div class="hint">no rollout videos available yet</div>';
  }

  // (d) per-step action viewer
  const seeds=Object.keys(D.traj).map(Number).sort((a,b)=>a-b);
  const sel=document.getElementById('seedSel');
  const armSel=document.getElementById('armSel');
  const note=document.getElementById('trajNote');
  seeds.forEach(s=>{const o=document.createElement('option');o.value=s;o.textContent=s;sel.appendChild(o);});
  const COLORS=['#1f78b4','#33a02c','#e31a1c','#ff7f00','#6a3d9a','#b15928','#a6cee3','#000000'];
  const LBL=['j0','j1','j2','j3','j4','j5','j6','gripper'];
  let actChart=null;
  function draw(){
    if(!seeds.length){note.textContent='no trajectory data yet';return;}
    const s=Number(sel.value), arm='arm'+armSel.value;
    const t=D.traj[s]; if(!t){note.textContent='no data for seed '+s;return;}
    const series=t[arm]||[];
    const dims=series.length?series[0].length:0;
    const ds=[];
    for(let k=0;k<dims;k++){
      ds.push({label:LBL[k]||('d'+k),
        data:series.map(row=>row[k]),
        borderColor:COLORS[k%COLORS.length],borderWidth:1.2,
        pointRadius:0,tension:.1});
    }
    if(actChart)actChart.destroy();
    actChart=new Chart(document.getElementById('actChart'),{
      type:'line',
      data:{labels:t.steps,datasets:ds},
      options:{responsive:true,maintainAspectRatio:false,
        scales:{x:{title:{display:true,text:'env step'}},
                y:{title:{display:true,text:'applied action (joint target / gripper)'}}},
        plugins:{legend:{labels:{boxWidth:12,font:{size:9}}}}}
    });
    note.textContent='seed '+s+' · '+t.n_steps+' env-steps ('+
      t.steps.length+' points shown after downsampling)';
  }
  sel.onchange=draw;armSel.onchange=draw;
  if(seeds.length){sel.value=seeds[0];}
  draw();
})();
""".strip()

    metrics = [
        {"k": "Eval", "v": f"2SC decent wc · step {CKPT_STEP}",
         "d": f"job {JOB_ID} · TwoRobotsStackCube · TABLE scene · 60 seeds (100-159)"},
        {"k": "Progress", "v": f"{n}/{TOTAL_SEEDS}",
         "d": state_txt},
        {"k": "Success rate", "v": f"{n_succ}/{n} ({sr:.1f}%)" if n else "0/0 (n/a)",
         "d": f"baseline to beat: {BASELINE_TXT}"},
        {"k": "Checkpoint", "v": f"step {CKPT_STEP} / {CKPT_TOTAL}",
         "d": "PRELIMINARY — intermediate checkpoint, model ~40% trained"},
    ]

    conclusion = (
        f"PRELIMINARY step-{CKPT_STEP} eval ({state_txt}). "
        f"SR so far {n_succ}/{n} ({sr:.1f}%) vs baseline {BASELINE_TXT}. "
        f"Intermediate checkpoint (~40% trained) — not a final result."
    )

    return {
        "title": "Pi0.5 2SC decent wc — proprio-dropout step-8k eval (60 seeds, TABLE)",
        "kind": "agent",
        "status": "in_progress",
        "conclusion": conclusion,
        "metrics": metrics,
        "visual": {"kind": "sandbox", "html": html, "js": js, "css": css},
    }


# ----------------------------------------------------------------------------
# field-notes HTTP I/O
# ----------------------------------------------------------------------------
def _req(method: str, url: str, payload: dict | None) -> dict:
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Content-Type", "application/json")
    req.add_header("X-Field-Notes-Key", FN_KEY)
    with urllib.request.urlopen(req, timeout=120) as resp:
        body = resp.read().decode()
        return json.loads(body) if body else {}


def cell_exists(cell_id: str) -> bool:
    try:
        _req("GET", f"{FN_BASE}/cells/{cell_id}", None)
        return True
    except Exception:
        return False


def create_cell(cell: dict) -> str:
    payload = dict(cell)
    payload["after_cell_id"] = AFTER_CELL_ID
    out = _req("POST", f"{FN_BASE}/projects/{PROJECT_ID}/cells", payload)
    cid = out.get("id")
    if not cid:
        raise RuntimeError(f"create returned no id: {out}")
    return cid


def update_cell(cell_id: str, cell: dict) -> None:
    # the PATCH endpoint rejects `kind` (extra_forbidden) — it is a
    # create-only field. Strip it before updating.
    payload = {k: v for k, v in cell.items() if k != "kind"}
    _req("PATCH", f"{FN_BASE}/cells/{cell_id}", payload)


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
def state_signature(results: list[dict], finished: bool) -> str:
    """Cheap fingerprint of the eval STATE (inputs), computed before any
    expensive ffmpeg work. x264 output is non-deterministic, so we must
    detect change from inputs — not from rendered output.

    Inputs that matter: which seeds completed (+success/steps), whether the
    job finished, the set of video files, and the trajectory file size."""
    parts = [f"finished={finished}"]
    for r in results:
        parts.append(f"{r['seed']}:{int(r['success'])}:{r['steps']}")
    if VIDEO_DIR.exists():
        vids = sorted(p.name for p in VIDEO_DIR.glob("*.mp4"))
        parts.append("vids=" + ",".join(vids))
    if TRAJ_PATH.exists():
        parts.append(f"trajsize={TRAJ_PATH.stat().st_size}")
    return hashlib.sha256("|".join(parts).encode()).hexdigest()


def main() -> int:
    results, finished = collect_results()
    running = job_running()
    log(f"results={len(results)} finished={finished} running={running}")

    cell_id = CELL_ID_FILE.read_text().strip() if CELL_ID_FILE.exists() else ""
    have_cell = bool(cell_id) and cell_exists(cell_id)

    # cheap no-op guard BEFORE transcoding: skip everything if the eval state
    # is unchanged since the last successful render.
    sig = state_signature(results, finished)
    prev = CONTENT_HASH_FILE.read_text().strip() if CONTENT_HASH_FILE.exists() else ""
    if have_cell and sig == prev:
        log("no change in eval state since last render — skipping")
        print(cell_id)
        return 0

    # state changed (or first run) — do the expensive work.
    traj = load_trajectories()
    gallery = build_gallery(results)
    cell = build_cell(results, traj, gallery, finished, running)

    if have_cell:
        update_cell(cell_id, cell)
        log(f"updated cell {cell_id}")
    else:
        cell_id = create_cell(cell)
        CELL_ID_FILE.write_text(cell_id)
        log(f"created cell {cell_id}")

    CONTENT_HASH_FILE.write_text(sig)
    print(cell_id)  # stdout: the cell id, for the caller
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        log(f"FATAL: {type(e).__name__}: {e}")
        sys.exit(0)  # never kill the polling loop
