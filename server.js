const express = require("express");
const path = require("path");
const fs = require("fs");
const fsp = require("fs/promises");
const { spawn } = require("child_process");
const { parse } = require("csv-parse/sync");
const archiver = require("archiver");

const app = express();
const PORT = process.env.PORT || 3000;

const WORKSPACE_ROOT = __dirname;
const PYTHON_SCRIPT = path.join(WORKSPACE_ROOT, "compare_mobile_models.py");
const RESULTS_DIR = path.join(WORKSPACE_ROOT, "results");
const PLOTS_DIR = path.join(RESULTS_DIR, "plots");
const HISTORY_FILE = path.join(RESULTS_DIR, "run_history.json");

const defaultPython = process.platform === "win32"
    ? path.join(WORKSPACE_ROOT, ".venv", "Scripts", "python.exe")
    : path.join(WORKSPACE_ROOT, ".venv", "bin", "python");
const PYTHON_EXECUTABLE = process.env.PYTHON_EXECUTABLE || defaultPython;

app.use(express.json({ limit: "1mb" }));
app.use(express.static(path.join(WORKSPACE_ROOT, "public")));
app.use("/results", express.static(RESULTS_DIR));

const runs = new Map();
const runProcesses = new Map();
let runHistory = [];

function sanitizeState(state) {
    return {
        runId: state.runId,
        status: state.status,
        startedAt: state.startedAt,
        finishedAt: state.finishedAt,
        code: state.code,
        args: state.args,
        config: state.config,
        error: state.error,
        canceled: !!state.canceled,
        pid: state.pid || null,
    };
}

async function loadRunHistory() {
    try {
        await fsp.mkdir(RESULTS_DIR, { recursive: true });
        if (!fs.existsSync(HISTORY_FILE)) {
            runHistory = [];
            return;
        }
        const raw = await fsp.readFile(HISTORY_FILE, "utf-8");
        const parsed = JSON.parse(raw);
        runHistory = Array.isArray(parsed) ? parsed : [];
    } catch (error) {
        runHistory = [];
        console.error("Failed to load run history:", error.message);
    }
}

async function persistRunHistory() {
    await fsp.mkdir(RESULTS_DIR, { recursive: true });
    await fsp.writeFile(HISTORY_FILE, JSON.stringify(runHistory, null, 2), "utf-8");
}

async function upsertRunHistory(state) {
    const compact = sanitizeState(state);
    const idx = runHistory.findIndex((r) => r.runId === compact.runId);
    if (idx >= 0) {
        runHistory[idx] = compact;
    } else {
        runHistory.unshift(compact);
    }
    runHistory = runHistory.slice(0, 100);
    await persistRunHistory();
}

function toArgs(cfg) {
    const args = [PYTHON_SCRIPT];

    const add = (flag, value) => {
        if (value === undefined || value === null || value === "") {
            return;
        }
        args.push(flag, String(value));
    };

    add("--epochs", cfg.epochs);
    add("--batch-size", cfg.batchSize);
    add("--lr", cfg.lr);
    add("--weight-decay", cfg.weightDecay);
    add("--img-size", cfg.imgSize);
    add("--benchmark-runs", cfg.benchmarkRuns);
    add("--workers", cfg.workers);

    if (cfg.distill) {
        args.push("--distill");
        add("--teacher-model", cfg.teacherModel || "efficientnet_b0");
        add("--distill-alpha", cfg.distillAlpha);
        add("--distill-temperature", cfg.distillTemperature);
    }

    if (cfg.skipQuant) {
        args.push("--skip-quant");
    }

    if (cfg.exportOnnx) {
        args.push("--export-onnx");
    }

    if (cfg.benchmarkOnnx) {
        args.push("--benchmark-onnx");
    }

    add("--onnx-opset", cfg.onnxOpset);

    return args;
}

function makeRunId() {
    return `run_${Date.now()}_${Math.floor(Math.random() * 10000)}`;
}

app.get("/api/health", (req, res) => {
    res.json({ ok: true, pythonExecutable: PYTHON_EXECUTABLE, script: PYTHON_SCRIPT });
});

app.post("/api/run", (req, res) => {
    if (!fs.existsSync(PYTHON_SCRIPT)) {
        return res.status(500).json({ error: "Python script not found." });
    }

    const runId = makeRunId();
    const cfg = req.body || {};
    const args = toArgs(cfg);

    const child = spawn(PYTHON_EXECUTABLE, args, {
        cwd: WORKSPACE_ROOT,
        shell: false,
        windowsHide: true,
    });

    const state = {
        runId,
        status: "running",
        startedAt: new Date().toISOString(),
        finishedAt: null,
        code: null,
        args,
        config: cfg,
        logs: "",
        error: null,
        canceled: false,
        pid: child.pid || null,
    };

    runs.set(runId, state);
    runProcesses.set(runId, child);
    upsertRunHistory(state).catch(() => { });

    child.stdout.on("data", (chunk) => {
        state.logs += chunk.toString();
    });

    child.stderr.on("data", (chunk) => {
        state.logs += chunk.toString();
    });

    child.on("error", (err) => {
        state.status = "failed";
        state.finishedAt = new Date().toISOString();
        state.error = err.message;
        runProcesses.delete(runId);
        upsertRunHistory(state).catch(() => { });
    });

    child.on("close", (code) => {
        state.code = code;
        state.finishedAt = new Date().toISOString();
        if (state.canceled) {
            state.status = "canceled";
        } else {
            state.status = code === 0 ? "completed" : "failed";
        }
        runProcesses.delete(runId);
        upsertRunHistory(state).catch(() => { });
    });

    res.status(202).json({ runId, status: state.status, args });
});

app.get("/api/run/:runId", (req, res) => {
    const state = runs.get(req.params.runId);
    if (!state) {
        return res.status(404).json({ error: "Run not found." });
    }
    return res.json(state);
});

app.post("/api/run/:runId/cancel", async (req, res) => {
    const runId = req.params.runId;
    const state = runs.get(runId);
    if (!state) {
        return res.status(404).json({ error: "Run not found." });
    }

    if (state.status !== "running") {
        return res.status(409).json({ error: `Run is already ${state.status}.` });
    }

    const child = runProcesses.get(runId);
    if (!child) {
        return res.status(409).json({ error: "Run process not available." });
    }

    state.canceled = true;
    state.logs += "\n[server] Cancellation requested by user.\n";
    const signal = process.platform === "win32" ? "SIGTERM" : "SIGINT";
    child.kill(signal);
    await upsertRunHistory(state);

    return res.json({ runId, status: "cancel-requested" });
});

app.get("/api/runs", (req, res) => {
    const active = Array.from(runs.values()).map((r) => sanitizeState(r));
    res.json({ history: runHistory, active });
});

app.get("/api/results", async (req, res) => {
    const csvPath = path.join(RESULTS_DIR, "model_comparison.csv");
    const mdPath = path.join(RESULTS_DIR, "model_comparison.md");

    const response = {
        hasResults: false,
        table: [],
        markdown: "",
        plots: [],
        files: {
            csv: fs.existsSync(csvPath) ? "/results/model_comparison.csv" : null,
            markdown: fs.existsSync(mdPath) ? "/results/model_comparison.md" : null,
        },
    };

    if (fs.existsSync(csvPath)) {
        const raw = await fsp.readFile(csvPath, "utf-8");
        response.table = parse(raw, { columns: true, skip_empty_lines: true });
        response.hasResults = true;
    }

    if (fs.existsSync(mdPath)) {
        response.markdown = await fsp.readFile(mdPath, "utf-8");
    }

    if (fs.existsSync(PLOTS_DIR)) {
        const files = await fsp.readdir(PLOTS_DIR);
        response.plots = files
            .filter((f) => f.toLowerCase().endsWith(".png"))
            .sort()
            .map((f) => ({
                name: f,
                url: `/results/plots/${encodeURIComponent(f)}`,
            }));
    }

    return res.json(response);
});

app.get("/api/export-bundle", async (req, res) => {
    await fsp.mkdir(RESULTS_DIR, { recursive: true });

    const timestamp = new Date().toISOString().replace(/[:.]/g, "-");
    const filename = `mobile-model-comparison-bundle-${timestamp}.zip`;
    res.setHeader("Content-Type", "application/zip");
    res.setHeader("Content-Disposition", `attachment; filename="${filename}"`);

    const archive = archiver("zip", { zlib: { level: 9 } });
    archive.on("error", (err) => {
        if (!res.headersSent) {
            res.status(500).json({ error: err.message });
        } else {
            res.end();
        }
    });

    archive.pipe(res);

    const csvPath = path.join(RESULTS_DIR, "model_comparison.csv");
    const mdPath = path.join(RESULTS_DIR, "model_comparison.md");
    if (fs.existsSync(csvPath)) {
        archive.file(csvPath, { name: "results/model_comparison.csv" });
    }
    if (fs.existsSync(mdPath)) {
        archive.file(mdPath, { name: "results/model_comparison.md" });
    }
    if (fs.existsSync(HISTORY_FILE)) {
        archive.file(HISTORY_FILE, { name: "results/run_history.json" });
    }
    if (fs.existsSync(PLOTS_DIR)) {
        archive.directory(PLOTS_DIR, "results/plots");
    }

    const onnxDir = path.join(RESULTS_DIR, "onnx");
    if (fs.existsSync(onnxDir)) {
        archive.directory(onnxDir, "results/onnx");
    }

    await archive.finalize();
});

loadRunHistory().then(() => {
    app.listen(PORT, () => {
        console.log(`Server running on http://localhost:${PORT}`);
        console.log(`Using python: ${PYTHON_EXECUTABLE}`);
    });
});
