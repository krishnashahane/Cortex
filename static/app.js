const $ = (s) => document.querySelector(s);
const api = (p, o) => fetch(p, o).then((r) => r.json());
let selected = null;
let pollTimer = null;
let lastEventId = 0;

const FLOW = ["CEO", "PaperReader", "HypothesisGenerator", "ExperimentPlanner", "Trainer", "Evaluator", "Critic", "ReportWriter"];

async function boot() {
  const c = await api("/api/config");
  $("#provider").textContent = "llm: " + c.llm_provider;
  $("#ds").textContent = "dataset: " + c.dataset;
  $("#thr").textContent = "stop < " + (c.improvement_threshold * 100).toFixed(1) + "%";
  $("#maxit").value = c.max_iterations;
  await refreshRuns();
}

async function refreshRuns() {
  const { runs, active } = await api("/api/runs");
  const el = $("#runs");
  el.innerHTML = "";
  if (!runs.length) el.innerHTML = '<p class="muted">No runs yet.</p>';
  for (const r of runs) {
    const d = document.createElement("div");
    d.className = "run" + (r.run_id === selected ? " sel" : "");
    const live = active.includes(r.run_id);
    d.innerHTML = `<div class="id">${r.run_id} ${live ? "● live" : ""}</div>
      <div class="meta">${r.status} · best ${(r.best_score || 0).toFixed(4)} · ${r.iterations || 0} iters</div>`;
    d.onclick = () => select(r.run_id);
    el.appendChild(d);
  }
}

$("#start").onclick = async () => {
  $("#start").disabled = true;
  const body = JSON.stringify({
    goal: $("#goal").value,
    max_iterations: parseInt($("#maxit").value) || null,
  });
  const r = await api("/api/runs", { method: "POST", headers: { "Content-Type": "application/json" }, body });
  $("#start").disabled = false;
  await refreshRuns();
  select(r.run_id);
};

function select(id) {
  selected = id;
  lastEventId = 0;
  if (pollTimer) clearInterval(pollTimer);
  refreshRuns();
  renderShell();
  poll();
  pollTimer = setInterval(poll, 1200);
}

function renderShell() {
  $("#detail").innerHTML = `
    <h2>Run ${selected}</h2>
    <div class="flow" id="flow">${FLOW.map((n) => `<span class="node" data-n="${n}">${n}</span>`).join("")}</div>
    <div class="grid4" id="stats"></div>
    <div class="card"><h2>Live agent feed</h2><div class="feed" id="feed"></div></div>
    <div class="card"><h2>Experiments</h2><div id="exps"></div></div>
    <div class="card"><h2>Report</h2><div id="reportbox"><span class="muted">Pending…</span></div></div>`;
}

async function poll() {
  if (!selected) return;
  const [run, exps, evd] = await Promise.all([
    api(`/api/runs/${selected}`),
    api(`/api/runs/${selected}/experiments`),
    api(`/api/runs/${selected}/events?after=${lastEventId}`),
  ]);
  renderStats(run, exps.experiments);
  renderExps(exps.experiments, run.best_experiment_id);
  appendEvents(evd.events);

  if (!evd.active && run.status === "completed") {
    clearInterval(pollTimer);
    pollTimer = null;
    loadReport();
    refreshRuns();
  }
}

function renderStats(run, exps) {
  const best = Math.max(0, run.best_score || 0);
  const lastImp = exps.length ? "" : "";
  $("#stats").innerHTML = `
    <div class="stat"><div class="k">Status</div><div class="v">${run.is_active ? "running" : run.status}</div></div>
    <div class="stat"><div class="k">Iterations</div><div class="v">${exps.length}</div></div>
    <div class="stat"><div class="k">Best score</div><div class="v">${best.toFixed(4)}</div></div>
    <div class="stat"><div class="k">Best model</div><div class="v" style="font-size:15px">${bestModel(exps, run.best_experiment_id)}</div></div>`;
}

function bestModel(exps, id) {
  const e = exps.find((x) => x.id === id) || exps.slice().sort((a, b) => b.score - a.score)[0];
  return e ? e.config.model : "—";
}

function renderExps(exps, bestId) {
  if (!exps.length) { $("#exps").innerHTML = '<span class="muted">No experiments yet.</span>'; return; }
  const rows = exps.map((e) => `
    <tr class="${e.id === bestId ? "best" : ""}">
      <td>${e.iteration}</td><td>${e.config.model}</td>
      <td>${(e.score || 0).toFixed(4)}</td>
      <td>${Object.entries(e.metrics || {}).map(([k, v]) => k + "=" + v.toFixed(3)).join(" ")}</td>
      <td>${(e.train_seconds || 0).toFixed(3)}s</td><td>${e.status}</td>
    </tr>`).join("");
  $("#exps").innerHTML = `<table><thead><tr><th>Iter</th><th>Model</th><th>Score</th><th>Metrics</th><th>Train</th><th>Status</th></tr></thead><tbody>${rows}</tbody></table>`;
}

function appendEvents(events) {
  if (!events.length) return;
  const feed = $("#feed");
  for (const e of events) {
    lastEventId = Math.max(lastEventId, e.id);
    const div = document.createElement("div");
    div.className = "ev kind-" + e.kind;
    div.innerHTML = `<span class="ag">${e.agent}</span><span class="msg">${escapeHtml(e.message)}</span>`;
    feed.appendChild(div);
    highlight(e.agent);
  }
  feed.scrollTop = feed.scrollHeight;
}

function highlight(agent) {
  document.querySelectorAll(".node").forEach((n) => n.classList.toggle("live", n.dataset.n === agent));
}

async function loadReport() {
  try {
    const md = await fetch(`/api/runs/${selected}/report`).then((r) => (r.ok ? r.text() : ""));
    $("#reportbox").innerHTML = md ? `<pre>${escapeHtml(md)}</pre>` : '<span class="muted">No report.</span>';
  } catch { /* ignore */ }
}

function escapeHtml(s) {
  return (s || "").replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));
}

boot();
