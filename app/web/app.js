const KEY_STORAGE = "cancerhawk.openrouter_key";
const PREFS_STORAGE = "cancerhawk.prefs";
const SIDEBAR_STORAGE = "cancerhawk.sidebar_collapsed";
const ROLES = ["submitter", "validator", "compiler", "archetype", "topic_deriver"];

// DOM elements
const apiKeyEl = document.getElementById("api_key");
const rememberEl = document.getElementById("remember");
const goalEl = document.getElementById("research_goal");
const nSubEl = document.getElementById("n_submitters");
const autoPubEl = document.getElementById("auto_publish");
const gitPushEl = document.getElementById("git_push");
const runBtn = document.getElementById("run_btn");
const progressSection = document.getElementById("progress_section");
const logEl = document.getElementById("log");
const resultSection = document.getElementById("result_section");
const resultSummary = document.getElementById("result_summary");
const resultLink = document.getElementById("result_link");
const localPreview = document.getElementById("local_preview");
const previewPaper = document.getElementById("preview_paper");
const previewPeer = document.getElementById("preview_peer");
const previewSimulations = document.getElementById("preview_simulations");

// Stats dashboard elements
const statCalls = document.getElementById("stat_calls");
const statTokens = document.getElementById("stat_tokens");
const statPhase = document.getElementById("stat_phase");
const statSuccess = document.getElementById("stat_success");
const statLatency = document.getElementById("stat_latency");
const statCost = document.getElementById("stat_cost");

// Totals elements
const tCalls = document.getElementById("t_calls");
const tIn = document.getElementById("t_in");
const tOut = document.getElementById("t_out");
const tTotal = document.getElementById("t_total");
const tCost = document.getElementById("t_cost");
const tElapsed = document.getElementById("t_elapsed");
const callCountSpan = document.getElementById("call_count");

// Sidebar elements
const sidebar = document.getElementById("sidebar");
const sidebarToggle = document.getElementById("sidebar_toggle");
const modeBadge = document.getElementById("mode_badge");
const researchTimer = document.getElementById("research_timer");
const sidebarIn = document.getElementById("sidebar_in");
const sidebarOut = document.getElementById("sidebar_out");
const sidebarTotal = document.getElementById("sidebar_total");
const modelBreakdown = document.getElementById("model_breakdown");
const modelBreakdownHeader = document.getElementById("model_breakdown_header");
const apiStats = document.getElementById("api_stats");
const apiStatsHeader = document.getElementById("api_stats_header");
const apiTotalCalls = document.getElementById("api_total_calls");
const apiSuccess = document.getElementById("api_success");
const apiFailed = document.getElementById("api_failed");
const apiSuccessRate = document.getElementById("api_success_rate");

// Summary row
const summaryTotal = document.getElementById("summary_total");
const summarySuccess = document.getElementById("summary_success");
const summaryLatency = document.getElementById("summary_latency");
const summaryCost = document.getElementById("summary_cost");

// Calls table
const callsTbody = document.getElementById("calls_tbody");

// Tab switching
const tabBtns = document.querySelectorAll(".tab");
const panes = document.querySelectorAll(".tab-pane");

// State
let ws = null;
let stats = {
  total_calls: 0,
  total_input: 0,
  total_output: 0,
  total_tokens: 0,
  total_cost_usd: 0,
  failed_calls: 0,
  elapsed_seconds: 0,
  by_model: {},
  by_role: {},
};
let currentPhase = "idle";
let runStartTime = 0;
let timerInterval = null;
let expandedRow = null;

function loadPrefs() {
  const storedApiKey = localStorage.getItem(KEY_STORAGE);
  if (storedApiKey) {
    apiKeyEl.value = storedApiKey;
    rememberEl.checked = true;
  }
  try {
    const prefs = JSON.parse(localStorage.getItem(PREFS_STORAGE) || "{}");
    if (prefs.research_goal) goalEl.value = prefs.research_goal;
    if (prefs.n_submitters) nSubEl.value = prefs.n_submitters;
    if (typeof prefs.auto_publish === "boolean") autoPubEl.checked = prefs.auto_publish;
    gitPushEl.checked = typeof prefs.git_push === "boolean" ? prefs.git_push : true;
    return prefs;
  } catch {
    return {};
  }
}

function savePrefs(prefs) {
  if (rememberEl.checked) localStorage.setItem(KEY_STORAGE, apiKeyEl.value.trim());
  else localStorage.removeItem(KEY_STORAGE);
  localStorage.setItem(PREFS_STORAGE, JSON.stringify(prefs));
}

function loadSidebarState() {
  const collapsed = localStorage.getItem(SIDEBAR_STORAGE) === "true";
  if (collapsed) sidebar.classList.add("collapsed");
  return !collapsed;
}

function saveSidebarState(collapsed) {
  localStorage.setItem(SIDEBAR_STORAGE, collapsed ? "true" : "false");
}

function startTimer() {
  runStartTime = Date.now();
  timerInterval = setInterval(() => {
    const elapsed = Math.floor((Date.now() - runStartTime) / 1000);
    const h = String(Math.floor(elapsed / 3600)).padStart(2, "0");
    const m = String(Math.floor((elapsed % 3600) / 60)).padStart(2, "0");
    const s = String(elapsed % 60).padStart(2, "0");
    researchTimer.textContent = `${h}:${m}:${s}`;
    tElapsed.textContent = `${elapsed}s`;
  }, 1000);
}

function stopTimer() {
  if (timerInterval) clearInterval(timerInterval);
  timerInterval = null;
}

async function loadModels() {
  const r = await fetch("/api/models");
  const { models, defaults } = await r.json();
  const prefs = loadPrefs();
  for (const role of ROLES) {
    const sel = document.getElementById(role);
    sel.innerHTML = "";
    for (const m of models) {
      const opt = document.createElement("option");
      opt.value = m;
      opt.textContent = m;
      sel.appendChild(opt);
    }
    sel.value = prefs[role] || defaults[role] || models[0];
  }
}

function appendLog(stage, message) {
  const row = document.createElement("div");
  row.className = "entry";
  row.innerHTML = `<span class="stage ${stage}">${stage}</span><span class="msg"></span>`;
  row.querySelector(".msg").textContent = message;
  logEl.appendChild(row);
  logEl.scrollTop = logEl.scrollHeight;
}

function updateStatsDashboard(stats) {
  statCalls.textContent = stats.total_calls.toLocaleString();
  statTokens.textContent = stats.total_tokens.toLocaleString();
  statPhase.textContent = currentPhase;
  const successRate = stats.total_calls > 0
    ? ((stats.total_calls - stats.failed_calls) / stats.total_calls * 100).toFixed(1) + "%"
    : "—";
  statSuccess.textContent = successRate;
  statLatency.textContent = stats.avg_latency_ms ? `${stats.avg_latency_ms}ms` : "—";
  statCost.textContent = `$${stats.total_cost_usd.toFixed(4)}`;
}

function updateTotals(stats) {
  tCalls.textContent = stats.total_calls;
  tIn.textContent = stats.total_input.toLocaleString();
  tOut.textContent = stats.total_output.toLocaleString();
  tTotal.textContent = stats.total_tokens.toLocaleString();
  tCost.textContent = `$${stats.total_cost_usd.toFixed(4)}`;
  callCountSpan.textContent = stats.total_calls;

  // Sidebar totals
  sidebarIn.textContent = stats.total_input.toLocaleString();
  sidebarOut.textContent = stats.total_output.toLocaleString();
  sidebarTotal.textContent = stats.total_tokens.toLocaleString();

  // API stats
  apiTotalCalls.textContent = stats.total_calls;
  apiSuccess.textContent = stats.total_calls - stats.failed_calls;
  apiFailed.textContent = stats.failed_calls;
  const rate = stats.total_calls > 0
    ? ((stats.total_calls - stats.failed_calls) / stats.total_calls * 100).toFixed(1) + "%"
    : "—";
  apiSuccessRate.textContent = rate;

  // Summary row
  summaryTotal.textContent = `${stats.total_calls} calls`;
  summarySuccess.textContent = `${rate} success`;
  summaryCost.textContent = `$${stats.total_cost_usd.toFixed(4)}`;
}

function updateModelBreakdown(stats) {
  const models = Object.entries(stats.by_model)
    .sort((a, b) => (b[1].input + b[1].output) - (a[1].input + a[1].output));

  modelBreakdown.innerHTML = models.map(([model, data]) => {
    const escapedModel = escapeHtml(model.split('/').pop());
    return `
      <div class="model-breakdown-item">
        <span class="model-name" title="${escapeHtml(model)}">${escapedModel}</span>
        <span class="model-tokens">${(data.input + data.output).toLocaleString()}</span>
      </div>
    `;
  }).join("");
}

function setPhase(phase) {
  currentPhase = phase;
  const labels = {
    starting: "Starting",
    paper_compile: "Paper",
    analysis: "Analysis",
    topic_derive: "Topics",
    peer_review: "Peer Review",
    publish: "Publish",
    done: "Done",
    error: "Error",
    idle: "Idle",
  };
  modeBadge.textContent = labels[phase] || phase;
  updateStatsDashboard(stats);
}

// Toggle row expansion for calls table
function toggleRow(tr) {
  // Check if there's already an expanded row following this one
  const next = tr.nextElementSibling;
  const isExpanded = next && next.classList.contains("expanded-row");

  // Collapse any other expanded rows
  const existing = document.querySelector(".expanded-row");
  if (existing && existing !== next) {
    existing.remove();
    existing.previousElementSibling?.classList.remove("expanded");
  }

  if (isExpanded) {
    // Collapse this row
    next.remove();
    tr.classList.remove("expanded");
    expandedRow = null;
  } else {
    // Expand this row
    const call = tr.dataset.call;
    if (call) {
      const data = JSON.parse(call);
      const expandTr = document.createElement("tr");
      expandTr.className = "expanded-row";
      const promptText = data.prompt ? data.prompt : '';
      const responseText = data.response ? data.response : '';
      expandTr.innerHTML = `
        <td colspan="8" class="expand-cell">
          <div class="expand-content">
            <pre><span class="label">PROMPT</span>${escapeHtml(promptText)}</pre>
            <pre><span class="label">RESPONSE</span>${escapeHtml(responseText)}</pre>
            <button class="copy-btn">Copy</button>
          </div>
        </td>
      `;
      const copyBtn = expandTr.querySelector(".copy-btn");
      copyBtn.onclick = (e) => {
        e.stopPropagation();
        copyToClipboard(promptText + "\n\n" + responseText, copyBtn);
      };
      tr.after(expandTr);
    }
    tr.classList.add("expanded");
    expandedRow = tr;
  }
}

function escapeHtml(str) {
  if (!str) return "";
  return String(str).replace(/[&<>"']/g, c => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
  }[c]));
}

async function copyToClipboard(text, btn) {
  try {
    await navigator.clipboard.writeText(text);
    btn.textContent = "Copied!";
    btn.classList.add("copied");
    setTimeout(() => {
      btn.textContent = "Copy";
      btn.classList.remove("copied");
    }, 1500);
  } catch (err) {
    btn.textContent = "Failed";
  }
}

function addCallRow(call) {
  const tr = document.createElement("tr");
  tr.dataset.call = JSON.stringify(call);
  tr.style.cursor = "pointer";
  tr.title = "Click to expand prompt/response";
  tr.onclick = () => toggleRow(tr);

  const okClass = call.ok ? "ok" : "fail";
  const okText = call.ok ? "✓" : "✗";
  const errorText = call.error ? ` title="${escapeHtml(call.error)}"` : "";

  // Escape all user/LLM-provided strings before injecting into HTML
  const roleBadge = escapeHtml(call.role);
  const modelShort = escapeHtml(call.model.split('/').pop());
  const modelTitle = escapeHtml(call.model);

  tr.innerHTML = `
    <td class="role"><span class="role-badge">${roleBadge}</span></td>
    <td class="model"><span class="model-badge" title="${modelTitle}">${modelShort}</span></td>
    <td>${call.prompt_tokens.toLocaleString()}</td>
    <td>${call.completion_tokens.toLocaleString()}</td>
    <td>${(call.prompt_tokens + call.completion_tokens).toLocaleString()}</td>
    <td>${call.latency_ms}ms</td>
    <td>$${call.cost_usd.toFixed(6)}</td>
    <td class="${okClass}"${errorText}>${okText}</td>
  `;
  callsTbody.appendChild(tr);
  // Scroll calls tab to bottom
  const container = document.querySelector("#calls");
  if (container) container.scrollTop = container.scrollHeight;

  // Apply active filters to new row
  applyFilters();

  // Update summary stats
  const avgLatency = stats.avg_latency_ms || 0;
  summaryLatency.textContent = `~${avgLatency}ms avg`;
}

function buildConfig() {
  const cfg = {
    api_key: apiKeyEl.value.trim(),
    research_goal: goalEl.value.trim(),
    n_submitters: parseInt(nSubEl.value, 10) || 3,
    auto_publish: autoPubEl.checked,
    git_push: gitPushEl.checked,
  };
  for (const role of ROLES) cfg[role] = document.getElementById(role).value;
  return cfg;
}

function resetUI() {
  tCalls.textContent = "0";
  tIn.textContent = "0";
  tOut.textContent = "0";
  tTotal.textContent = "0";
  tCost.textContent = "$0.00000";
  tElapsed.textContent = "0s";
  callCountSpan.textContent = "0";
  callsTbody.innerHTML = "";
  statCalls.textContent = "0";
  statTokens.textContent = "0";
  statPhase.textContent = "idle";
  statSuccess.textContent = "—";
  statLatency.textContent = "—";
  statCost.textContent = "$0.00";
  sidebarIn.textContent = "0";
  sidebarOut.textContent = "0";
  sidebarTotal.textContent = "0";
  modelBreakdown.innerHTML = "";
  apiTotalCalls.textContent = "0";
  apiSuccess.textContent = "0";
  apiFailed.textContent = "0";
  apiSuccessRate.textContent = "—";
  summaryTotal.textContent = "0 calls";
  summarySuccess.textContent = "0% success";
  summaryLatency.textContent = "~0ms avg";
  summaryCost.textContent = "$0.0000";
  modeBadge.textContent = "Idle";
  researchTimer.textContent = "00:00:00";
  localPreview.hidden = true;
  previewPaper.innerHTML = "";
  previewPeer.innerHTML = "";
  previewSimulations.innerHTML = "";
  stopTimer();

  // Reset filters
  const roleFilter = document.getElementById("filter_role");
  const statusFilter = document.getElementById("filter_status");
  if (roleFilter) roleFilter.value = "all";
  if (statusFilter) statusFilter.value = "all";
}

async function run() {
  const cfg = buildConfig();
  if (!cfg.api_key) {
    alert("OpenRouter API key required");
    return;
  }
  if (!cfg.research_goal) {
    alert("Research goal required");
    return;
  }
  if (cfg.research_goal.length > 1000) {
    alert("Research goal must be at most 1000 characters");
    return;
  }

  const prefsToSave = { ...cfg };
  delete prefsToSave.api_key;
  savePrefs(prefsToSave);

  resetUI();
  runBtn.disabled = true;
  runBtn.textContent = "▶ Running…";
  progressSection.hidden = false;
  resultSection.hidden = true;
  logEl.innerHTML = "";

  // Start health polling
  const healthPoll = setInterval(async () => {
    try {
      const r = await fetch("/api/health");
      const data = await r.json();
      if (data.status === "ok") {
        // Server reachable — local timer is authoritative
      }
    } catch (e) {}
  }, 3000);

  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  ws = new WebSocket(`${proto}//${location.host}/ws/hermes/run`);
  
  ws.onopen = () => {
    ws.send(JSON.stringify(cfg));
    startTimer();
    setPhase("starting");
  };

  ws.onmessage = (e) => {
    let evt;
    try { evt = JSON.parse(e.data); } catch { appendLog("error", e.data); return; }
    
    const stage = evt.stage || "info";
    appendLog(stage, evt.message || "");

    // Update phase badge
    const phaseMap = {
      start: "starting",
      paper_done: "paper_compile",
      analyze: "analysis",
      derive: "topic_derive",
      simulate: "peer_review",
      simulate_done: "peer_review",
      publish: "publish",
      publish_done: "publish",
      review: "peer_review",
      review_complete: "peer_review",
      done: "done",
      error: "error",
    };
    if (phaseMap[stage]) setPhase(phaseMap[stage]);

    if (stage === "api_call" && evt.data) {
      const call = evt.data.call;
      const totals = evt.data.totals;
      stats = totals;
      addCallRow(call);
      updateTotals(stats);
      updateModelBreakdown(stats);
      updateStatsDashboard(stats);
    }

    if (stage === "done") {
      setPhase("done");
      stopTimer();
      clearInterval(healthPoll);
      const url = evt.data?.result_url;
      const price = evt.data?.market_price;
      resultSummary.textContent =
        `✓ Block ${evt.data?.block ?? "?"} complete · "${evt.data?.title ?? ""}" · ` +
        `synthesis market price = ${(price * 100).toFixed(0)}%`;
      if (url) {
        resultLink.href = url.startsWith("/") ? url : "/" + url;
        resultLink.hidden = false;
      } else {
        resultLink.hidden = true;
      }
      resultSection.hidden = false;
      if (evt.data?.block) {
        loadBlockPreview(evt.data.block).catch(err => {
          appendLog("error", `local preview failed: ${err.message || err}`);
        });
      }
    }
  };

  ws.onerror = () => appendLog("error", "WebSocket error");
  ws.onclose = () => {
    runBtn.disabled = false;
    runBtn.textContent = "▶ Run a fresh CancerHawk block";
    stopTimer();
    clearInterval(healthPoll);
  };
}

// Tab switching
tabBtns.forEach(btn => {
  btn.addEventListener("click", () => {
    tabBtns.forEach(b => b.classList.remove("active"));
    panes.forEach(p => p.classList.remove("active"));
    btn.classList.add("active");
    const tabId = btn.getAttribute("data-tab");
    document.getElementById(tabId).classList.add("active");
  });
});

document.querySelectorAll(".result-tab").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".result-tab").forEach(b => b.classList.remove("active"));
    document.querySelectorAll(".result-pane").forEach(p => p.classList.remove("active"));
    btn.classList.add("active");
    document.getElementById(`preview_${btn.dataset.resultTab}`).classList.add("active");
  });
});

// Sidebar toggle
sidebarToggle.addEventListener("click", () => {
  const isCollapsed = sidebar.classList.toggle("collapsed");
  saveSidebarState(isCollapsed);
});

// Collapsible sections in sidebar
modelBreakdownHeader.addEventListener("click", () => {
  const section = modelBreakdownHeader.parentElement;
  section.classList.toggle("collapsed");
});
apiStatsHeader.addEventListener("click", () => {
  const section = apiStatsHeader.parentElement;
  section.classList.toggle("collapsed");
});

// Initialize sidebar state
loadSidebarState();

// Event listeners
runBtn.addEventListener("click", run);
loadModels();
setupFilters();

// Filtering
function applyFilters() {
  const roleSelect = document.getElementById("filter_role");
  const statusSelect = document.getElementById("filter_status");
  if (!roleSelect || !statusSelect) return;
  const roleFilter = roleSelect.value;
  const statusFilter = statusSelect.value;
  const rows = callsTbody.querySelectorAll("tr:not(.expanded-row)");
  rows.forEach(row => {
    const roleCell = row.querySelector(".role");
    const statusCell = row.querySelector("td:last-child");
    const rowRole = roleCell ? roleCell.textContent.trim().toLowerCase() : "";
    const rowStatus = statusCell ? (statusCell.classList.contains("ok") ? "ok" : "fail") : "";
    let show = true;
    if (roleFilter !== "all" && rowRole !== roleFilter) show = false;
    if (statusFilter !== "all" && rowStatus !== statusFilter) show = false;
    row.style.display = show ? "" : "none";

    // Also hide the expanded row if parent is hidden
    const next = row.nextElementSibling;
    if (next && next.classList.contains("expanded-row")) {
      next.style.display = show ? "" : "none";
    }
  });
}

function setupFilters() {
  const roleSelect = document.getElementById("filter_role");
  const statusSelect = document.getElementById("filter_status");
  if (!roleSelect || !statusSelect) return;
  roleSelect.addEventListener("change", applyFilters);
  statusSelect.addEventListener("change", applyFilters);
  applyFilters(); // initial
}

async function loadBlockPreview(blockNumber) {
  localPreview.hidden = false;
  previewPaper.innerHTML = `<div class="preview-loading">Loading generated paper…</div>`;
  previewPeer.innerHTML = `<div class="preview-loading">Loading peer review…</div>`;
  previewSimulations.innerHTML = `<div class="preview-loading">Loading simulations…</div>`;

  const response = await fetch(`/api/blocks/${blockNumber}`);
  if (!response.ok) {
    throw new Error(`Block ${blockNumber} bundle returned ${response.status}`);
  }
  const bundle = await response.json();
  renderPaperPreview(bundle.paper_md || "");
  renderPeerReviewPreview(bundle.peer_reviews || []);
  renderSimulationPreview(bundle.simulations || []);
}

function renderPaperPreview(markdown) {
  if (!markdown.trim()) {
    previewPaper.innerHTML = `<p class="muted">No generated paper was found for this block.</p>`;
    return;
  }
  const lines = markdown.split(/\r?\n/);
  const html = [];
  let paragraph = [];

  function flushParagraph() {
    if (!paragraph.length) return;
    html.push(`<p>${escapeHtml(paragraph.join(" "))}</p>`);
    paragraph = [];
  }

  for (const line of lines) {
    const trimmed = line.trim();
    if (!trimmed) {
      flushParagraph();
      continue;
    }
    const heading = trimmed.match(/^(#{1,4})\s+(.+)$/);
    if (heading) {
      flushParagraph();
      const level = Math.min(heading[1].length + 2, 5);
      html.push(`<h${level}>${escapeHtml(heading[2])}</h${level}>`);
      continue;
    }
    paragraph.push(trimmed.replace(/\*\*/g, ""));
  }
  flushParagraph();
  previewPaper.innerHTML = `<article class="paper-preview">${html.join("")}</article>`;
}

function renderPeerReviewPreview(reviews) {
  if (!reviews.length) {
    previewPeer.innerHTML = `<p class="muted">No peer-review records were found for this block.</p>`;
    return;
  }
  previewPeer.innerHTML = reviews.map(review => {
    const concerns = Array.isArray(review.criticisms) ? review.criticisms : [];
    const fixes = Array.isArray(review.actionable_fixes) ? review.actionable_fixes : [];
    const experiments = Array.isArray(review.suggested_experiments) ? review.suggested_experiments : [];
    return `
      <article class="review-card">
        <div class="review-card__top">
          <strong>${escapeHtml(review.archetype || "Peer Reviewer")}</strong>
          <span>${escapeHtml(review.recommendation || "reviewed")}</span>
        </div>
        <p>${escapeHtml(review.summary || "No summary provided.")}</p>
        ${renderMiniList("Key concerns", concerns)}
        ${renderMiniList("Fixes requested", fixes)}
        ${renderMiniList("Suggested experiments", experiments)}
      </article>
    `;
  }).join("");
}

function renderMiniList(title, items) {
  if (!items.length) return "";
  return `
    <div class="mini-list">
      <h4>${escapeHtml(title)}</h4>
      <ul>${items.slice(0, 4).map(item => `<li>${escapeHtml(item)}</li>`).join("")}</ul>
    </div>
  `;
}

function renderSimulationPreview(simulations) {
  if (!simulations.length) {
    previewSimulations.innerHTML = `<p class="muted">No generated simulations were found for this block.</p>`;
    return;
  }
  previewSimulations.innerHTML = simulations.map((sim, index) => `
    <article class="simulation-card">
      <div class="simulation-card__copy">
        <span class="sim-chip">${escapeHtml(sim.type || "html5_canvas")}</span>
        <h3>${escapeHtml(sim.title || `Simulation ${index + 1}`)}</h3>
        <p>${escapeHtml(sim.description || sim.rationale || "Generated from the peer-reviewed paper.")}</p>
      </div>
      <canvas class="local-sim-canvas" width="720" height="320" data-sim-index="${index}" aria-label="${escapeHtml(sim.title || "CancerHawk simulation")}"></canvas>
    </article>
  `).join("");
  bootLocalSimulationCanvases();
}

function bootLocalSimulationCanvases() {
  const canvases = previewSimulations.querySelectorAll(".local-sim-canvas");
  canvases.forEach(canvas => {
    const ctx = canvas.getContext("2d");
    const index = Number(canvas.dataset.simIndex || 0);
    const cells = Array.from({ length: 34 + index * 5 }, (_, i) => ({
      x: (i * 53 + index * 37) % canvas.width,
      y: (i * 31 + index * 59) % canvas.height,
      r: 4 + ((i + index) % 7),
      vx: 0.25 + ((i + index) % 5) * 0.06,
      vy: 0.16 + ((i * 3 + index) % 5) * 0.05,
      phase: i * 0.31,
    }));

    function draw(time) {
      if (!canvas.isConnected) return;
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      const gradient = ctx.createRadialGradient(120, 40, 20, canvas.width / 2, canvas.height / 2, canvas.width);
      gradient.addColorStop(0, "#14351e");
      gradient.addColorStop(1, "#050a06");
      ctx.fillStyle = gradient;
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      for (const cell of cells) {
        cell.x = (cell.x + cell.vx) % canvas.width;
        cell.y = (cell.y + cell.vy) % canvas.height;
        const pulse = Math.sin(time * 0.002 + cell.phase) * 1.8;
        ctx.beginPath();
        ctx.arc(cell.x, cell.y, Math.max(2, cell.r + pulse), 0, Math.PI * 2);
        ctx.fillStyle = index % 2 ? "rgba(111, 219, 111, 0.68)" : "rgba(79, 195, 247, 0.58)";
        ctx.fill();
        ctx.strokeStyle = "rgba(255, 255, 255, 0.18)";
        ctx.stroke();
      }

      ctx.strokeStyle = "rgba(201, 162, 39, 0.32)";
      ctx.lineWidth = 1;
      for (let i = 0; i < cells.length - 1; i += 3) {
        const a = cells[i];
        const b = cells[(i + 7 + index) % cells.length];
        ctx.beginPath();
        ctx.moveTo(a.x, a.y);
        ctx.lineTo(b.x, b.y);
        ctx.stroke();
      }

      requestAnimationFrame(draw);
    }
    requestAnimationFrame(draw);
  });
}
