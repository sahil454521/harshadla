const healthBadge = document.getElementById("healthBadge");
const refreshBtn = document.getElementById("refreshBtn");
const exportBtn = document.getElementById("exportBtn");
const cancelBtn = document.getElementById("cancelBtn");
const runForm = document.getElementById("runForm");
const runMeta = document.getElementById("runMeta");
const runLogs = document.getElementById("runLogs");
const tableContainer = document.getElementById("tableContainer");
const plotsContainer = document.getElementById("plotsContainer");
const historyContainer = document.getElementById("historyContainer");

let activeRunId = null;
let pollTimer = null;

function updateRunControls(isRunning) {
    cancelBtn.disabled = !isRunning;
}

function setBadge(text, cls = "") {
    healthBadge.textContent = text;
    healthBadge.className = `badge ${cls}`.trim();
}

function formDataToPayload(form) {
    const fd = new FormData(form);
    const getNum = (k) => {
        const value = fd.get(k);
        return value === null || value === "" ? null : Number(value);
    };

    return {
        epochs: getNum("epochs"),
        batchSize: getNum("batchSize"),
        lr: getNum("lr"),
        weightDecay: getNum("weightDecay"),
        imgSize: getNum("imgSize"),
        benchmarkRuns: getNum("benchmarkRuns"),
        workers: getNum("workers"),
        onnxOpset: getNum("onnxOpset"),
        distill: fd.get("distill") === "on",
        skipQuant: fd.get("skipQuant") === "on",
        exportOnnx: fd.get("exportOnnx") === "on",
        benchmarkOnnx: fd.get("benchmarkOnnx") === "on",
        teacherModel: fd.get("teacherModel"),
        distillAlpha: getNum("distillAlpha"),
        distillTemperature: getNum("distillTemperature"),
    };
}

async function checkHealth() {
    try {
        const res = await fetch("/api/health");
        if (!res.ok) {
            throw new Error(`HTTP ${res.status}`);
        }
        const data = await res.json();
        setBadge(`Server OK | Python: ${data.pythonExecutable}`, "ok");
    } catch (err) {
        setBadge(`Server error: ${err.message}`, "err");
    }
}

function renderTable(rows) {
    if (!rows || rows.length === 0) {
        tableContainer.textContent = "No results available yet.";
        return;
    }

    const cols = Object.keys(rows[0]);
    const head = `<thead><tr>${cols.map((c) => `<th>${c}</th>`).join("")}</tr></thead>`;
    const body = rows
        .map((row) => `<tr>${cols.map((c) => `<td>${row[c] ?? ""}</td>`).join("")}</tr>`)
        .join("");
    tableContainer.innerHTML = `<table>${head}<tbody>${body}</tbody></table>`;
}

function renderPlots(plots) {
    if (!plots || plots.length === 0) {
        plotsContainer.textContent = "No plots yet.";
        return;
    }

    plotsContainer.innerHTML = plots
        .map(
            (p) => `
      <article class="plot-card">
        <img src="${p.url}?t=${Date.now()}" alt="${p.name}" />
        <p>${p.name}</p>
      </article>
    `
        )
        .join("");
}

function renderRunHistory(history) {
    if (!history || history.length === 0) {
        historyContainer.textContent = "No run history yet.";
        return;
    }

    historyContainer.innerHTML = history
        .map((r) => {
            const statusClass = String(r.status || "").toLowerCase();
            const argsTxt = (r.args || []).join(" ");
            return `
                <article class="history-item">
                    <div class="row">
                        <strong>${r.runId}</strong>
                        <span class="pill ${statusClass}">${r.status}</span>
                        <span>code: ${r.code ?? "-"}</span>
                    </div>
                    <div class="row">
                        <span>start: ${r.startedAt || "-"}</span>
                        <span>finish: ${r.finishedAt || "-"}</span>
                    </div>
                    <div class="row">
                        <span>args: ${argsTxt || "-"}</span>
                    </div>
                </article>
            `;
        })
        .join("");
}

async function loadRunHistory() {
    try {
        const res = await fetch("/api/runs");
        if (!res.ok) {
            throw new Error(`HTTP ${res.status}`);
        }
        const data = await res.json();
        const merged = [...(data.active || []), ...(data.history || [])];
        const dedup = [];
        const seen = new Set();
        for (const item of merged) {
            if (!seen.has(item.runId)) {
                dedup.push(item);
                seen.add(item.runId);
            }
        }
        renderRunHistory(dedup);
    } catch (err) {
        historyContainer.textContent = `Failed to load run history: ${err.message}`;
    }
}

async function loadResults() {
    try {
        const res = await fetch("/api/results");
        const data = await res.json();
        renderTable(data.table || []);
        renderPlots(data.plots || []);
    } catch (err) {
        tableContainer.textContent = `Failed to load results: ${err.message}`;
    }

    await loadRunHistory();
}

async function pollRun(runId) {
    try {
        const res = await fetch(`/api/run/${runId}`);
        if (!res.ok) {
            throw new Error(`HTTP ${res.status}`);
        }

        const data = await res.json();
        runMeta.textContent = `Run ${data.runId} | status: ${data.status} | started: ${data.startedAt}`;
        runLogs.textContent = data.logs || "No logs yet.";
        runLogs.scrollTop = runLogs.scrollHeight;

        if (data.status === "completed" || data.status === "failed") {
            clearInterval(pollTimer);
            pollTimer = null;
            updateRunControls(false);
            await loadResults();
        } else if (data.status === "canceled") {
            clearInterval(pollTimer);
            pollTimer = null;
            updateRunControls(false);
            await loadResults();
        } else {
            updateRunControls(true);
        }
    } catch (err) {
        runMeta.textContent = `Polling failed: ${err.message}`;
        if (pollTimer) {
            clearInterval(pollTimer);
            pollTimer = null;
        }
        updateRunControls(false);
    }
}

runForm.addEventListener("submit", async (evt) => {
    evt.preventDefault();
    const payload = formDataToPayload(runForm);
    runMeta.textContent = "Starting run...";
    runLogs.textContent = "";
    updateRunControls(false);

    try {
        const res = await fetch("/api/run", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
        });

        if (!res.ok) {
            const msg = await res.text();
            throw new Error(msg || `HTTP ${res.status}`);
        }

        const data = await res.json();
        activeRunId = data.runId;
        runMeta.textContent = `Run ${activeRunId} started.`;
        updateRunControls(true);

        if (pollTimer) {
            clearInterval(pollTimer);
        }
        pollTimer = setInterval(() => pollRun(activeRunId), 2000);
        pollRun(activeRunId);
    } catch (err) {
        runMeta.textContent = `Failed to start run: ${err.message}`;
        updateRunControls(false);
    }
});

refreshBtn.addEventListener("click", loadResults);

cancelBtn.addEventListener("click", async () => {
    if (!activeRunId) {
        runMeta.textContent = "No active run to cancel.";
        return;
    }

    try {
        const res = await fetch(`/api/run/${activeRunId}/cancel`, { method: "POST" });
        if (!res.ok) {
            const txt = await res.text();
            throw new Error(txt || `HTTP ${res.status}`);
        }
        runMeta.textContent = `Cancel requested for run ${activeRunId}.`;
    } catch (err) {
        runMeta.textContent = `Cancel failed: ${err.message}`;
    }
});

exportBtn.addEventListener("click", () => {
    const link = document.createElement("a");
    link.href = `/api/export-bundle?t=${Date.now()}`;
    link.download = "mobile-model-comparison-bundle.zip";
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
});

checkHealth();
updateRunControls(false);
loadResults();
