/* app.js — TumorEvo frontend */

// ── State ─────────────────────────────────────────────────────────
const state = {
  schedule:      "continuous",
  r:             0.3,
  mu:            1e-6,
  kill_eff:      0.8,
  treat_start:   0,
  results:       null,
  metrics:       null,
  spatialFrames: null,
};

let debounceTimer = null;
let animTimer     = null;
let animDay       = 0;
let isAnimating   = false;

const COL = {
  S:   "#1B3D8F",
  RA:  "#B03A2E",
  RB:  "#1B6B3A",
  RAB: "#C47A00",
  tox: "#9A6B00",
};

// RGBA pixel values per cell state (0=empty, 1=S, 2=RA, 3=RB, 4=RAB)
const CELL_RGBA = [
  [10,  10,  10,  255],
  [27,  61,  143, 255],
  [176, 58,  46,  255],
  [27,  107, 58,  255],
  [196, 122, 0,   255],
];

const BASE_LAYOUT = {
  paper_bgcolor: "rgba(0,0,0,0)",
  plot_bgcolor:  "#FEFCF8",
  font:   { family: "IBM Plex Mono", size: 10, color: "#444444" },
  margin: { l: 52, r: 14, t: 8, b: 36 },
  xaxis: {
    gridcolor: "#E8E5DF", linecolor: "#111", tickcolor: "#111",
    zeroline: false, title: { text: "day", standoff: 6 },
  },
  yaxis: {
    gridcolor: "#E8E5DF", linecolor: "#111", tickcolor: "#111",
    zeroline: false,
  },
  showlegend: true,
  legend: { orientation: "h", y: -0.22, font: { size: 9 } },
  hoverlabel: {
    bgcolor: "#FFFFFF", bordercolor: "#CCCCCC",
    font: { family: "IBM Plex Mono", size: 11, color: "#111" },
  },
};

const PLOTLY_CONFIG = { responsive: true, displayModeBar: false };

// ── API ───────────────────────────────────────────────────────────
async function fetchSimulation() {
  const res = await fetch("/api/simulate", {
    method:  "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      r: state.r, mu: state.mu,
      kill_eff: state.kill_eff, treat_start: state.treat_start,
    }),
  });
  if (!res.ok) { const e = await res.json(); throw new Error(e.error || "failed"); }
  return res.json();
}

// ── Main run ──────────────────────────────────────────────────────
async function runAll() {
  stopAnim();
  showSpinner("solving odes…");
  setStatus("running");
  try {
    const data = await fetchSimulation();
    state.results = data.results;
    state.metrics = data.metrics;

    showSpinner("building spatial model…");
    await microtask();
    state.spatialFrames = buildSpatialFrames(state.results[state.schedule]);

    renderBurden();
    renderFractions();
    renderToxicity();
    renderSpatialFrame(state.spatialFrames.length - 1);
    updateClinical();
    updateStats();
    checkWarnings();
    setStatus("ready");
  } catch (e) {
    setStatus("error");
    showWarn("Error: " + e.message);
    console.error(e);
  } finally {
    hideSpinner();
  }
}

// ── SPATIAL CA ────────────────────────────────────────────────────
/*
 * The grid is ODE-driven: total cell count and type fractions are
 * computed from the ODE solution at each day, then the CA runs only
 * to generate spatial clustering (not to track populations).
 *
 * This guarantees the grid never goes black: if the ODE says N=1e8
 * the grid will always show cells, even mid-treatment when drug is
 * wiping out sensitive cells and resistant clones are small.
 */
const GRID  = 50;
const NCELL = GRID * GRID;

// Map N → grid fill using log scale so small populations are visible.
// N=1e6 (initial seed) → ~15% fill.  N=K=1e10 → 85% fill.
function nToFill(N) {
  const logN = Math.log10(Math.max(1, N));
  const logK = 10; // log10(1e10)
  return Math.max(0.04, Math.min(0.85, (logN / logK) * 0.85));
}

function buildSpatialFrames(res) {
  const muVis = Math.min(0.06, state.mu * 2e5); // amplify so mutations are visible
  const DAYS  = 200;

  // Initialise grid: small S-cell disk at centre
  const grid = new Uint8Array(NCELL);
  const cx = Math.floor(GRID / 2), cy = Math.floor(GRID / 2);
  for (let dy = -3; dy <= 3; dy++)
    for (let dx = -3; dx <= 3; dx++)
      if (dx*dx + dy*dy <= 9) grid[(cy+dy)*GRID+(cx+dx)] = 1;

  // Read ODE state at integer day d
  function odeAt(day) {
    const idx = Math.min(day * 10, res.t.length - 1);
    const N   = Math.max(0, res.N[idx]);
    const eps = 1e-12;
    return {
      N,
      fS:   res.S[idx]   / (N + eps),
      fRA:  res.RA[idx]  / (N + eps),
      fRB:  res.RB[idx]  / (N + eps),
      fRAB: res.RAB[idx] / (N + eps),
    };
  }

  const frames = [];

  for (let day = 0; day <= DAYS; day++) {
    // ── Step 1: hard-sync grid to ODE (guarantees no black box) ──
    const { N, fS, fRA, fRB, fRAB } = odeAt(day);
    const targetCount = Math.round(nToFill(N) * NCELL);
    syncPopulation(grid, targetCount, fS, fRA, fRB, fRAB);

    // ── Step 2: save frame after sync ────────────────────────────
    frames.push(grid.slice());

    // ── Step 3: CA diffusion pass — spreads clusters spatially ───
    // (runs AFTER save so frame reflects ODE state, not CA drift)
    caPass(grid, muVis, state.r);
  }

  return frames;
}

/*
 * syncPopulation — the key fix.
 *
 * Adjusts the grid so:
 *   - total living cells = targetCount
 *   - each type's share matches the ODE fractions
 *
 * Order of operations:
 *   1. Set total count (add/remove cells preserving clusters)
 *   2. Re-type cells to match fS/fRA/fRB/fRAB
 */
function syncPopulation(grid, targetCount, fS, fRA, fRB, fRAB) {
  const counts = countByType(grid);
  const living = counts[1] + counts[2] + counts[3] + counts[4];

  // ── Adjust total count ────────────────────────────────────────
  if (living < targetCount) {
    const toAdd = targetCount - living;
    addCells(grid, toAdd, fS, fRA, fRB, fRAB);
  } else if (living > targetCount) {
    const toRemove = living - targetCount;
    removeCells(grid, toRemove, fS);
  }

  // ── Re-type to match fractions ────────────────────────────────
  retypeCells(grid, fS, fRA, fRB, fRAB);
}

/*
 * addCells: place new cells adjacent to existing ones (keeps clusters),
 * falling back to random placement if grid is empty.
 */
function addCells(grid, count, fS, fRA, fRB, fRAB) {
  let added = 0;
  // Two passes: first near existing cells, then random
  for (let pass = 0; pass < 2 && added < count; pass++) {
    for (let i = 0; i < NCELL && added < count; i++) {
      if (pass === 0 && grid[i] === 0) continue; // first pass: must be occupied
      if (pass === 1 && grid[i] !== 0) continue; // second pass: must be empty

      let target;
      if (pass === 0) {
        // Place adjacent to cell i
        target = emptyNeighbour(grid, i);
      } else {
        // Random empty cell
        target = i;
      }

      if (target === -1 || grid[target] !== 0) continue;
      grid[target] = pickType(fS, fRA, fRB, fRAB);
      added++;
    }
  }
}

/*
 * removeCells: preferentially remove sensitive cells (drug kills them first),
 * then resistant, scattered to avoid destroying cluster coherence entirely.
 */
function removeCells(grid, count, fS) {
  // Priority: remove S cells first (drug effect), then others
  const removeOrder = fS > 0.5 ? [1, 2, 3, 4] : [1, 2, 3, 4];
  let removed = 0;
  for (const type of removeOrder) {
    for (let i = 0; i < NCELL && removed < count; i++) {
      if (grid[i] === type && Math.random() < 0.5) {
        grid[i] = 0;
        removed++;
      }
    }
    if (removed >= count) break;
  }
}

/*
 * retypeCells: convert cells so the type distribution matches ODE fractions.
 * Works by scanning the grid and converting the most over-represented type
 * into the most under-represented, in small batches to avoid jarring jumps.
 */
function retypeCells(grid, fS, fRA, fRB, fRAB) {
  const counts  = countByType(grid);
  const living  = counts[1] + counts[2] + counts[3] + counts[4];
  if (living === 0) return;

  const targets = [
    { type: 1, target: fS   * living },
    { type: 2, target: fRA  * living },
    { type: 3, target: fRB  * living },
    { type: 4, target: fRAB * living },
  ];

  // How many cells of each type need to change
  const delta = targets.map(t => ({
    type: t.type,
    diff: counts[t.type] - Math.round(t.target), // positive = too many, negative = too few
  }));

  // Convert over-represented → under-represented
  // Cap per-day conversion at 8% of total to keep transitions smooth
  const maxConvert = Math.ceil(living * 0.08);
  let converted = 0;

  // Build source list (types with excess) and sink list (types with deficit)
  const sources = delta.filter(d => d.diff > 2).sort((a, b) => b.diff - a.diff);
  const sinks   = delta.filter(d => d.diff < -2).sort((a, b) => a.diff - b.diff);

  if (sources.length === 0 || sinks.length === 0) return;

  for (let i = 0; i < NCELL && converted < maxConvert; i++) {
    const src = sources.find(s => s.diff > 2);
    const snk = sinks.find(s => s.diff < -2);
    if (!src || !snk) break;

    if (grid[i] === src.type && Math.random() < 0.3) {
      grid[i] = snk.type;
      src.diff--;
      snk.diff++;
      converted++;
    }
  }
}

// Pick a random cell type weighted by ODE fractions
function pickType(fS, fRA, fRB, fRAB) {
  const r = Math.random();
  if (r < fS)              return 1;
  if (r < fS + fRA)        return 2;
  if (r < fS + fRA + fRB)  return 3;
  return 4;
}

/*
 * caPass: one round of spatial diffusion + mutation.
 * Cells divide into empty neighbours, mutate occasionally.
 * This creates the clustered appearance without controlling counts.
 */
function caPass(grid, muVis, r) {
  const pDiv = Math.min(0.35, r * 0.12);
  const order = shuffledIndices(NCELL);
  for (let ii = 0; ii < NCELL; ii++) {
    const i    = order[ii];
    const type = grid[i];
    if (type === 0 || Math.random() >= pDiv) continue;
    const nb = emptyNeighbour(grid, i);
    if (nb === -1) continue;
    let child = type;
    if      (type === 1) { const rv = Math.random(); if (rv < muVis) child = 2; else if (rv < 2*muVis) child = 3; }
    else if (type === 2 && Math.random() < muVis) child = 4;
    else if (type === 3 && Math.random() < muVis) child = 4;
    grid[nb] = child;
  }
}

function countByType(grid) {
  const c = [0,0,0,0,0];
  for (let i = 0; i < grid.length; i++) c[grid[i]]++;
  return c;
}
function emptyNeighbour(grid, i) {
  const row = Math.floor(i / GRID), col = i % GRID;
  const cands = [];
  for (let dr = -1; dr <= 1; dr++) for (let dc = -1; dc <= 1; dc++) {
    if (dr===0 && dc===0) continue;
    const nr = row+dr, nc = col+dc;
    if (nr<0||nr>=GRID||nc<0||nc>=GRID) continue;
    const ni = nr*GRID+nc;
    if (grid[ni] === 0) cands.push(ni);
  }
  return cands.length ? cands[Math.floor(Math.random()*cands.length)] : -1;
}
let _sBuf = null;
function shuffledIndices(n) {
  if (!_sBuf || _sBuf.length !== n) {
    _sBuf = new Int32Array(n);
    for (let i = 0; i < n; i++) _sBuf[i] = i;
  }
  for (let i = n-1; i > 0; i--) {
    const j = Math.floor(Math.random()*(i+1));
    const t = _sBuf[i]; _sBuf[i] = _sBuf[j]; _sBuf[j] = t;
  }
  return _sBuf;
}

// ── Render spatial frame ──────────────────────────────────────────
function renderSpatialFrame(day) {
  if (!state.spatialFrames) return;
  const grid   = state.spatialFrames[Math.min(day, state.spatialFrames.length-1)];
  const canvas = el("spatial-canvas");
  const ctx    = canvas.getContext("2d");
  const cw = canvas.width, ch = canvas.height;
  const cellW = cw/GRID, cellH = ch/GRID;

  const img  = ctx.createImageData(cw, ch);
  const data = img.data;

  for (let row = 0; row < GRID; row++) {
    for (let col = 0; col < GRID; col++) {
      const rgba = CELL_RGBA[grid[row*GRID+col]];
      const x0 = Math.floor(col*cellW), x1 = Math.floor((col+1)*cellW);
      const y0 = Math.floor(row*cellH), y1 = Math.floor((row+1)*cellH);
      for (let py = y0; py < y1; py++) {
        for (let px = x0; px < x1; px++) {
          const off = (py*cw+px)*4;
          data[off]=rgba[0]; data[off+1]=rgba[1]; data[off+2]=rgba[2]; data[off+3]=rgba[3];
        }
      }
    }
  }
  ctx.putImageData(img, 0, 0);

  el("spatial-day").textContent = `day ${day}`;
  const dose = getDoseAtDay(day);
  const doseEl = el("spatial-dose");
  if (day >= state.treat_start && dose > 0) {
    doseEl.textContent = `${(dose*100).toFixed(0)}% dose active`;
    doseEl.className = "mono dose-on";
  } else {
    doseEl.textContent = day < state.treat_start ? "pre-treatment" : "no treatment";
    doseEl.className = "mono dose-off";
  }
}

function getDoseAtDay(day) {
  if (day < state.treat_start) return 0;
  const e = day - state.treat_start;
  if (state.schedule === "continuous")  return 0.30;
  if (state.schedule === "pulsed")      return (e%7)<1 ? 1.0 : 0;
  if (state.schedule === "metronomic")  return (e%2.3)<0.5 ? 0.4 : 0;
  if (state.schedule === "escalating")  return Math.min(1.5, 0.5+e/100);
  return 0;
}

// ── Chart 1: Burden ───────────────────────────────────────────────
function renderBurden(sliceEnd) {
  const res = currentRes(sliceEnd);
  if (!res) return;
  const safeLog = v => v < 1 ? null : v;
  const traces = [
    { x:res.t, y:res.N, name:"N (linear)", yaxis:"y", mode:"lines",
      line:{color:"#111",width:1.5},
      hovertemplate:"day %{x:.1f}<br>N=%{y:.3e}<extra>linear</extra>" },
    { x:res.t, y:res.N.map(safeLog), name:"N (log₁₀)", yaxis:"y2", mode:"lines",
      line:{color:COL.S,width:1.5,dash:"dot"},
      hovertemplate:"day %{x:.1f}<br>N=%{y:.3e}<extra>log</extra>" },
  ];
  const layout = { ...BASE_LAYOUT, margin:{...BASE_LAYOUT.margin,r:56},
    yaxis:  { ...BASE_LAYOUT.yaxis, title:{text:"cells",standoff:4}, rangemode:"nonnegative" },
    yaxis2: { ...BASE_LAYOUT.yaxis, title:{text:"log₁₀",standoff:4},
              overlaying:"y", side:"right", type:"log", showgrid:false },
  };
  Plotly.react("chart-burden", traces, layout, PLOTLY_CONFIG);
}

// ── Chart 2: Fractions ────────────────────────────────────────────
function renderFractions(sliceEnd) {
  const res = currentRes(sliceEnd);
  if (!res) return;
  const eps = 1e-12;
  const frac = k => res[k].map((v,i) => (v/(res.N[i]+eps))*100);
  const mk = (key, name, col, fill) => ({
    x:res.t, y:frac(key), name, mode:"lines",
    fill, stackgroup:"one",
    fillcolor:col+"99", line:{color:col,width:0.5},
    hovertemplate:`day %{x:.1f}  ${name} = %{y:.1f}%<extra></extra>`,
  });
  const traces = [
    mk("S",   "S",    COL.S,   "tozeroy"),
    mk("RA",  "R_A",  COL.RA,  "tonexty"),
    mk("RB",  "R_B",  COL.RB,  "tonexty"),
    mk("RAB", "R_AB", COL.RAB, "tonexty"),
  ];
  const layout = { ...BASE_LAYOUT,
    yaxis:{ ...BASE_LAYOUT.yaxis, range:[0,100], title:{text:"%",standoff:4} } };
  Plotly.react("chart-fractions", traces, layout, PLOTLY_CONFIG);
}

// ── Chart 3: Toxicity ─────────────────────────────────────────────
function renderToxicity(sliceEnd) {
  const res = currentRes(sliceEnd);
  if (!res) return;
  const tEnd = res.t[res.t.length-1] || 200;
  const traces = [
    { x:[0,tEnd,tEnd,0], y:[80,80,108,108], fill:"toself", mode:"none",
      fillcolor:"rgba(176,58,46,0.06)", line:{color:"transparent"},
      name:"danger zone", hoverinfo:"none" },
    { x:[0,tEnd], y:[80,80], mode:"lines",
      line:{color:"rgba(176,58,46,0.4)",width:1,dash:"dash"},
      showlegend:false, hoverinfo:"none" },
    { x:res.t, y:res.T, name:"toxicity", mode:"lines",
      fill:"tozeroy", fillcolor:COL.tox+"18",
      line:{color:COL.tox,width:2},
      hovertemplate:"day %{x:.1f}<br>T=%{y:.1f}<extra></extra>" },
  ];
  const layout = { ...BASE_LAYOUT,
    yaxis:{ ...BASE_LAYOUT.yaxis, range:[0,108], title:{text:"index",standoff:4} } };
  Plotly.react("chart-toxicity", traces, layout, PLOTLY_CONFIG);
}

// ── Clinical ──────────────────────────────────────────────────────
function updateClinical() {
  const m = state.metrics?.[state.schedule];
  if (!m) return;
  const badge = el("outcome-badge");
  badge.textContent = m.outcome.toUpperCase();
  badge.className   = "outcome-badge " + m.outcome;
  el("clin-relapse").textContent = m.relapse_day != null ? `day ${m.relapse_day.toFixed(0)}` : "none";
  el("clin-tox").textContent = m.max_tox.toFixed(1);
  el("clin-ti").textContent  = m.ti.toFixed(3);
  el("rec-text").innerHTML   = buildRec(m);
}

function buildRec(m) {
  const sched  = state.schedule[0].toUpperCase() + state.schedule.slice(1);
  const muExp  = Math.log10(state.mu).toFixed(1);
  let txt = `<strong>${sched}</strong> · r=${state.r.toFixed(2)}/d · μ=10<sup>${muExp}</sup> · ${(state.kill_eff*100).toFixed(0)}% efficiency. `;
  if (m.outcome === "eradication") {
    txt += `Tumour burden falls below 1 cell — potentially curative. Monitor for minimal residual disease and consider de-escalation to limit long-term toxicity.`;
  } else if (m.outcome === "relapse") {
    txt += `Relapse predicted at day ${m.relapse_day?.toFixed(0)}. `;
    if (m.max_tox > 70) txt += `Toxicity (${m.max_tox.toFixed(1)}) limits escalation. `;
    txt += `R_AB clones are driving regrowth — consider adaptive therapy (Gatenby 2020) or dual-resistant subpopulation targeting.`;
  } else {
    txt += `Disease stabilises below 10⁷ cells through competitive suppression. `;
    txt += m.max_tox < 50
      ? `Low toxicity leaves headroom to intensify dosing if eradication is the goal.`
      : `Toxicity near maintenance limit; de-escalation may be warranted.`;
  }
  return txt;
}

function updateStats() {
  const m = state.metrics?.[state.schedule];
  if (!m) return;
  el("stat-peak").textContent    = fmtN(m.peak_N);
  el("stat-final").textContent   = fmtN(m.final_N);
  el("stat-tox").textContent     = m.max_tox.toFixed(1);
  el("stat-relapse").textContent = m.relapse_day != null ? `day ${m.relapse_day.toFixed(0)}` : "—";
}

function checkWarnings() {
  const m = state.metrics?.[state.schedule];
  if (!m) return;
  let msg = "";
  if (m.max_tox > 80)  msg = `Toxicity ${m.max_tox.toFixed(1)} exceeds danger threshold (80). Reduce dose intensity.`;
  if (m.peak_N > 1e13) msg = "Numerical instability: tumour greatly exceeds carrying capacity.";
  showWarn(msg);
}

// ── Animation ─────────────────────────────────────────────────────
function startAnim() {
  if (!state.spatialFrames) return;
  stopAnim();
  isAnimating = true;
  animDay = 0;
  const totalDays = state.spatialFrames.length;

  el("btn-anim").textContent = "PAUSE";
  el("btn-anim").classList.add("playing");
  el("anim-bar-wrap").removeAttribute("hidden");

  animTimer = setInterval(() => {
    animDay = Math.min(animDay + 1, totalDays - 1);
    el("anim-bar-fill").style.width = (animDay / (totalDays-1) * 100) + "%";

    renderSpatialFrame(animDay);
    if (animDay % 5 === 0) {
      const si = animDay * 10 + 1;
      renderBurden(si);
      renderFractions(si);
      renderToxicity(si);
    }
    if (animDay >= totalDays - 1) stopAnim();
  }, 1000 / 60);
}

function stopAnim() {
  if (animTimer) { clearInterval(animTimer); animTimer = null; }
  isAnimating = false;
  el("btn-anim").textContent = "ANIMATE";
  el("btn-anim").classList.remove("playing");
  el("anim-bar-wrap").setAttribute("hidden", "");
  el("anim-bar-fill").style.width = "0%";
  renderBurden(); renderFractions(); renderToxicity();
  if (state.spatialFrames) renderSpatialFrame(state.spatialFrames.length - 1);
}

function currentRes(end) {
  const res = state.results?.[state.schedule];
  if (!res || !end) return res;
  return {
    t: res.t.slice(0,end), S: res.S.slice(0,end), RA: res.RA.slice(0,end),
    RB: res.RB.slice(0,end), RAB: res.RAB.slice(0,end),
    N: res.N.slice(0,end), T: res.T.slice(0,end),
  };
}

// ── CSV ───────────────────────────────────────────────────────────
async function exportCSV() {
  const res = await fetch("/api/csv", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ r:state.r, mu:state.mu, kill_eff:state.kill_eff,
                           treat_start:state.treat_start, schedule:state.schedule }),
  });
  if (!res.ok) return;
  const blob = await res.blob();
  const cd   = res.headers.get("Content-Disposition") || "";
  const name = cd.split("filename=")[1] || "tumorevo.csv";
  const url  = URL.createObjectURL(blob);
  const a    = document.createElement("a");
  a.href = url; a.download = name; a.click(); URL.revokeObjectURL(url);
}

// ── Params ────────────────────────────────────────────────────────
function readParams() {
  state.r           = parseFloat(el("sl-r").value);
  state.mu          = Math.pow(10, parseFloat(el("sl-mu").value));
  state.kill_eff    = parseInt(el("sl-kill").value) / 100;
  state.treat_start = parseInt(el("sl-start").value);
  const muExp = Math.log10(state.mu).toFixed(1).replace(".0","");
  el("val-r").value     = state.r.toFixed(2) + " /d";
  el("val-mu").value    = "1e" + muExp;
  el("val-kill").value  = (state.kill_eff*100).toFixed(0) + "%";
  el("val-start").value = "day " + state.treat_start;
}

// ── Events ────────────────────────────────────────────────────────
document.querySelectorAll(".sched-btn").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".sched-btn").forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
    state.schedule = btn.dataset.sched;
    if (state.results) {
      showSpinner("rebuilding spatial model…");
      microtask().then(() => {
        state.spatialFrames = buildSpatialFrames(state.results[state.schedule]);
        renderBurden(); renderFractions(); renderToxicity();
        renderSpatialFrame(state.spatialFrames.length - 1);
        updateClinical(); updateStats(); checkWarnings();
        hideSpinner();
      });
    } else runAll();
  });
});

document.querySelectorAll("input[type=range]").forEach(sl => {
  sl.addEventListener("input", () => {
    readParams();
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(runAll, 300);
  });
});

el("btn-run").addEventListener("click", runAll);
el("btn-reset").addEventListener("click", () => {
  stopAnim();
  el("sl-r").value="0.3"; el("sl-mu").value="-6";
  el("sl-kill").value="80"; el("sl-start").value="0";
  document.querySelectorAll(".sched-btn").forEach(b => b.classList.remove("active"));
  document.querySelector('[data-sched="continuous"]').classList.add("active");
  state.schedule = "continuous";
  readParams(); runAll();
});
el("btn-anim").addEventListener("click", () => { if (isAnimating) stopAnim(); else startAnim(); });
el("btn-csv").addEventListener("click", exportCSV);
el("ref-btn").addEventListener("click", function() {
  const body = el("ref-body"), isOpen = body.hasAttribute("hidden");
  body.toggleAttribute("hidden", !isOpen);
  this.setAttribute("aria-expanded", isOpen);
});

// ── Helpers ───────────────────────────────────────────────────────
function el(id) { return document.getElementById(id); }
function fmtN(n) {
  if (n<1)    return "<1";
  if (n<1e3)  return n.toFixed(0);
  if (n<1e6)  return (n/1e3).toFixed(1)+"K";
  if (n<1e9)  return (n/1e6).toFixed(2)+"M";
  if (n<1e12) return (n/1e9).toFixed(2)+"B";
  return n.toExponential(2);
}
const microtask = () => new Promise(r => setTimeout(r, 20));
function showSpinner(msg) { el("spinner-msg").textContent=msg; el("spinner-overlay").classList.remove("hidden"); }
function hideSpinner()    { el("spinner-overlay").classList.add("hidden"); }
function setStatus(s)     { el("header-status").textContent = s; }
function showWarn(msg) {
  const bar = el("warn-bar");
  if (msg) { el("warn-text").textContent=msg; bar.removeAttribute("hidden"); }
  else       bar.setAttribute("hidden","");
}

// ── Init ──────────────────────────────────────────────────────────
readParams();
runAll();
