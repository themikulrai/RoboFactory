// Single-page RoboFactory dataset viewer.
// Loads manifest.json once; switches task -> repopulates list, re-renders plots,
// preloads next episode for zero-lag advance. Only 2 <video> elements are ever
// mounted (the active player + the hidden preloader), so there is no browser OOM
// risk even at 150 episodes per task.

"use strict";

const state = {
  manifest: null,           // {tasks: [...], episodes: [...]}
  byTask: {},               // task name -> [episode rows, unsorted]
  taskNames: [],
  currentTask: null,
  visibleEps: [],           // filtered+sorted view of episodes for current task
  currentEp: null,          // index into visibleEps
  stats: null,              // parsed stats/<task>.json
};

const el = {
  taskSelect: document.getElementById("task-select"),
  prevTask: document.getElementById("prev-task"),
  nextTask: document.getElementById("next-task"),
  taskBadges: document.getElementById("task-badges"),
  filterSuccess: document.getElementById("filter-success"),
  filterFailure: document.getElementById("filter-failure"),
  sortBy: document.getElementById("sort-by"),
  episodeList: document.getElementById("episode-list"),
  player: document.getElementById("player"),
  preloader: document.getElementById("preloader"),
  episodeTitle: document.getElementById("episode-title"),
  episodeMeta: document.getElementById("episode-meta"),
  prevEp: document.getElementById("prev-ep"),
  nextEp: document.getElementById("next-ep"),
  replayEp: document.getElementById("replay-ep"),
  statsSummary: document.getElementById("stats-summary"),
  figLength: document.getElementById("fig-length"),
  figJerk: document.getElementById("fig-jerk"),
  figSaturation: document.getElementById("fig-saturation"),
  figPathStd: document.getElementById("fig-path-std"),
};

async function init() {
  const manifest = await fetch("manifest.json", { cache: "no-cache" }).then(r => r.json()) /* no-cache so fresh stats/manifest are always fetched */;
  state.manifest = manifest;
  state.buildId = manifest.build_id || Date.now();
  state.taskNames = manifest.tasks.map(t => t.name);
  const bust = `?v=${state.buildId}`;
  for (const row of manifest.episodes) {
    row.video = row.video + bust;
    row.thumb = row.thumb + bust;
    if (!state.byTask[row.task]) state.byTask[row.task] = [];
    state.byTask[row.task].push(row);
  }
  for (const name of state.taskNames) {
    const opt = document.createElement("option");
    opt.value = name; opt.textContent = name;
    el.taskSelect.appendChild(opt);
  }
  wireEvents();
  await switchTask(state.taskNames[0]);
}

function wireEvents() {
  el.taskSelect.addEventListener("change", () => switchTask(el.taskSelect.value));
  el.prevTask.addEventListener("click", () => stepTask(-1));
  el.nextTask.addEventListener("click", () => stepTask(+1));
  el.prevEp.addEventListener("click", () => stepEpisode(-1));
  el.nextEp.addEventListener("click", () => stepEpisode(+1));
  el.replayEp.addEventListener("click", () => { el.player.currentTime = 0; el.player.play(); });
  el.filterSuccess.addEventListener("change", rebuildList);
  el.filterFailure.addEventListener("change", rebuildList);
  el.sortBy.addEventListener("change", rebuildList);

  document.addEventListener("keydown", (e) => {
    if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT") return;
    if (e.key === "ArrowRight") { stepEpisode(+1); e.preventDefault(); }
    else if (e.key === "ArrowLeft") { stepEpisode(-1); e.preventDefault(); }
    else if (e.key === "r") { el.player.currentTime = 0; el.player.play(); }
    else if (e.key === "t") { stepTask(+1); }
    else if (e.key === " ") {
      if (el.player.paused) el.player.play(); else el.player.pause();
      e.preventDefault();
    }
  });
}

async function switchTask(name) {
  state.currentTask = name;
  el.taskSelect.value = name;
  const taskMeta = state.manifest.tasks.find(t => t.name === name);
  el.taskBadges.textContent = `${taskMeta.n_agents} agent${taskMeta.n_agents > 1 ? "s" : ""} · ${taskMeta.n_episodes} eps · ${(taskMeta.success_rate * 100).toFixed(1)}% success`;

  state.stats = await fetch(`stats/${name}.json`, { cache: "no-cache" }).then(r => r.json()) /* no-cache so fresh stats/manifest are always fetched */;
  renderStatsSummary(state.stats);
  renderFigures(state.stats);

  rebuildList();
}

function stepTask(delta) {
  const i = state.taskNames.indexOf(state.currentTask);
  const j = (i + delta + state.taskNames.length) % state.taskNames.length;
  switchTask(state.taskNames[j]);
}

function rebuildList() {
  const rows = state.byTask[state.currentTask] || [];
  let view = rows.slice();
  if (el.filterSuccess.checked) view = view.filter(r => r.success);
  if (el.filterFailure.checked) view = view.filter(r => !r.success);
  const sortKey = el.sortBy.value;
  view.sort((a, b) => {
    if (sortKey === "ep") return a.ep - b.ep;
    if (sortKey === "length") return b.length - a.length;
    if (sortKey === "jerk") return b.jerk - a.jerk;
    return 0;
  });
  state.visibleEps = view;

  el.episodeList.innerHTML = "";
  view.forEach((r, i) => {
    const li = document.createElement("li");
    li.dataset.idx = i;
    li.innerHTML = `
      <img src="${r.thumb}" loading="lazy" />
      <div class="ep-meta">
        <div class="ep-num"><span class="succ-dot ${r.success ? "ok" : "fail"}"></span>ep ${r.ep}</div>
        <div class="ep-sub">len ${r.length} · jerk ${r.jerk.toFixed(3)}</div>
      </div>`;
    li.addEventListener("click", () => selectEpisode(i));
    el.episodeList.appendChild(li);
  });
  if (view.length > 0) selectEpisode(0);
  else { el.player.removeAttribute("src"); el.episodeTitle.textContent = "(no episodes)"; el.episodeMeta.textContent = ""; }
}

function selectEpisode(idx) {
  if (idx < 0 || idx >= state.visibleEps.length) return;
  state.currentEp = idx;
  const r = state.visibleEps[idx];

  // swap in preloaded next episode if present; else load fresh
  if (el.preloader.src && el.preloader.src.endsWith(r.video)) {
    el.player.src = el.preloader.src;
  } else {
    el.player.src = r.video;
  }
  el.player.load();
  el.player.play().catch(() => { /* autoplay may be blocked; user clicks play */ });

  el.episodeTitle.textContent = `${r.task} / ep ${r.ep}`;
  el.episodeMeta.textContent = `${r.success ? "✓" : "✗"} · ${r.length} steps · jerk ${r.jerk.toFixed(3)} · ${r.n_cams} cam${r.n_cams > 1 ? "s" : ""}`;

  // highlight active in list
  el.episodeList.querySelectorAll("li").forEach(li => li.classList.remove("active"));
  const active = el.episodeList.querySelector(`li[data-idx="${idx}"]`);
  if (active) {
    active.classList.add("active");
    active.scrollIntoView({ block: "nearest" });
  }

  // preload next
  const nextIdx = idx + 1;
  if (nextIdx < state.visibleEps.length) {
    el.preloader.src = state.visibleEps[nextIdx].video;
    el.preloader.load();
  } else {
    el.preloader.removeAttribute("src");
  }
}

function stepEpisode(delta) {
  if (state.currentEp == null) return;
  const next = state.currentEp + delta;
  if (next < 0 || next >= state.visibleEps.length) return;
  selectEpisode(next);
}

function renderStatsSummary(s) {
  const kv = [
    ["success rate", (s.success_rate * 100).toFixed(1) + "%"],
    ["episodes", s.n_episodes],
    ["action dim", s.n_action_dim],
    ["length mean", s.length_mean.toFixed(0)],
    ["length range", `${s.length_min}–${s.length_max}`],
    ["jerk mean", s.jerk_mean.toFixed(3)],
    ["scene diversity", s.diversity_l2.toFixed(2)],
    ["path diversity", (s.path_diversity_l2 ?? 0).toFixed(2)],
    ["max sat", Math.max(...s.action_saturation).toFixed(3)],
  ];
  el.statsSummary.innerHTML = kv
    .map(([k, v]) => `<div class="k">${k}</div><div class="v">${v}</div>`)
    .join("");
}

function renderFigures(s) {
  const cfg = { displayModeBar: false, responsive: true };
  const layoutTheme = {
    paper_bgcolor: "#13151b",
    plot_bgcolor: "#171a21",
    font: { color: "#cfd3da", size: 11 },
    xaxis: { gridcolor: "#232731" },
    yaxis: { gridcolor: "#232731" },
  };
  const apply = (fig) => ({
    data: fig.data,
    layout: { ...fig.layout, ...layoutTheme,
              xaxis: { ...(fig.layout.xaxis || {}), ...layoutTheme.xaxis },
              yaxis: { ...(fig.layout.yaxis || {}), ...layoutTheme.yaxis } },
  });
  const f = s.figures;
  const safeReact = (div, fig, name) => {
    if (!div) { console.warn(`[figure:${name}] div missing`); return; }
    if (!fig) { console.warn(`[figure:${name}] no figure data`); return; }
    try {
      const a = apply(fig);
      Plotly.react(div, a.data, a.layout, cfg);
    } catch (e) {
      console.error(`[figure:${name}] Plotly failed:`, e);
    }
  };
  safeReact(el.figLength, f.length, "length");
  safeReact(el.figJerk, f.jerk, "jerk");
  safeReact(el.figSaturation, f.saturation, "saturation");
  safeReact(el.figPathStd, f.path_std, "path_std");
}

init();
