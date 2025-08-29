// ========= Utility =========
const $ = (s) => document.querySelector(s);
const $$ = (s) => Array.from(document.querySelectorAll(s));

function rng(seed = 123) {
  let s = seed >>> 0;
  return function () {
    s ^= s << 13;
    s ^= s >>> 17;
    s ^= s << 5;
    s >>>= 0;
    return (s % 1_000_000) / 1_000_000;
  };
}
function shuffleInPlace(arr, rand) {
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
}

// ========= NN Core =========
const Activations = {
  relu: { f: (x) => Math.max(0, x), df: (y) => (y > 0 ? 1 : 0) },
  sigmoid: { f: (x) => 1 / (1 + Math.exp(-x)), df: (y) => y * (1 - y) },
  tanh: { f: (x) => Math.tanh(x), df: (y) => 1 - y * y },
  linear: { f: (x) => x, df: (_) => 1 },
};
function zeros(r, c) {
  return Array.from({ length: r }, () => Array(c).fill(0));
}
function randn(r, c, rand) {
  const m = zeros(r, c);
  for (let i = 0; i < r; i++)
    for (let j = 0; j < c; j++) {
      let u = 0,
        v = 0;
      while (u === 0) u = rand();
      while (v === 0) v = rand();
      const z = Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
      m[i][j] = z * 0.1;
    }
  return m;
}
function dot(a, b) {
  const r = a.length,
    c = b[0].length,
    n = b.length;
  const m = zeros(r, c);
  for (let i = 0; i < r; i++)
    for (let j = 0; j < c; j++) {
      let s = 0;
      for (let k = 0; k < n; k++) s += a[i][k] * b[k][j];
      m[i][j] = s;
    }
  return m;
}
function addBias(a, b) {
  const r = a.length,
    c = a[0].length;
  const m = zeros(r, c);
  for (let i = 0; i < r; i++)
    for (let j = 0; j < c; j++) m[i][j] = a[i][j] + b[0][j];
  return m;
}
function applyActivation(a, act) {
  const r = a.length,
    c = a[0].length;
  const m = zeros(r, c);
  for (let i = 0; i < r; i++)
    for (let j = 0; j < c; j++) m[i][j] = act.f(a[i][j]);
  return m;
}
function hadamard(a, b) {
  const r = a.length,
    c = a[0].length;
  const m = zeros(r, c);
  for (let i = 0; i < r; i++)
    for (let j = 0; j < c; j++) m[i][j] = a[i][j] * b[i][j];
  return m;
}
function transpose(a) {
  const r = a.length,
    c = a[0].length;
  const m = zeros(c, r);
  for (let i = 0; i < r; i++) for (let j = 0; j < c; j++) m[j][i] = a[i][j];
  return m;
}
function mse(pred, target) {
  const r = pred.length,
    c = pred[0].length;
  let sum = 0;
  const grad = zeros(r, c);
  for (let i = 0; i < r; i++)
    for (let j = 0; j < c; j++) {
      const e = pred[i][j] - target[i][j];
      sum += e * e;
      grad[i][j] = (2 * e) / c;
    }
  return [sum / r, grad];
}

class DenseLayer {
  constructor(
    inputSize,
    outputSize,
    activation = "relu",
    useBias = true,
    rand = Math.random
  ) {
    this.in = inputSize;
    this.out = outputSize;
    this.activation = activation;
    this.useBias = useBias;
    this.W = randn(inputSize, outputSize, rand);
    this.b = useBias ? zeros(1, outputSize) : null;
  }
  forward(X) {
    this.X = X;
    const Zlin = addBias(
      dot(X, this.W),
      this.useBias ? this.b : zeros(1, this.out)
    );
    this.Z = Zlin;
    this.A = applyActivation(Zlin, Activations[this.activation]);
    return this.A;
  }
  backward(dA, lr) {
    const act = Activations[this.activation];
    const r = this.A.length,
      c = this.A[0].length;
    const dZ = hadamard(
      dA,
      this.A.map((row) => row.map(act.df))
    );
    const Xt = transpose(this.X);
    const dW = dot(Xt, dZ);
    const dB = this.useBias
      ? [
          Array.from({ length: c }, (_, j) =>
            dZ.reduce((s, row) => s + row[j], 0)
          ),
        ]
      : null;
    const Wt = transpose(this.W);
    const dX = dot(dZ, Wt);
    for (let i = 0; i < this.W.length; i++)
      for (let j = 0; j < this.W[0].length; j++)
        this.W[i][j] -= (lr * dW[i][j]) / r;
    if (this.useBias)
      for (let j = 0; j < this.b[0].length; j++)
        this.b[0][j] -= (lr * dB[0][j]) / r;
    return dX;
  }
}
class Network {
  constructor() {
    this.layers = [];
  }
  add(L) {
    this.layers.push(L);
  }
  forward(X) {
    let A = X;
    for (const L of this.layers) A = L.forward(A);
    return A;
  }
  backward(dA, lr) {
    for (let i = this.layers.length - 1; i >= 0; i--)
      dA = this.layers[i].backward(dA, lr);
  }
}

// ========= State =========
let net = new Network();
let arch = []; // [{id,type:'input'|'hidden'|'output', neurons, activation, bias}]
let inputSize = 2,
  outputSize = 1;
let dataset = { X: [], y: [] };
let chart,
  stopFlag = false;
let lastNodeColors = null; // {byLayer:[], raw:[]}

// ========= Colors & Node Coloring =========
const clamp01 = (v) => Math.max(0, Math.min(1, v));
const lerp = (a, b, t) => a + (b - a) * t;
const hiddenColor = (v) =>
  `hsl(210 80% ${Math.round(lerp(12, 60, clamp01(v)))}%)`;
const outputColor = (v) => `hsl(${Math.round(120 * clamp01(v))} 85% 50%)`;

function computeNodeColorsForInput(xvec) {
  if (!xvec) return;
  net.forward([xvec]);
  lastNodeColors = { byLayer: [], raw: [] };

  net.layers.forEach((L, k) => {
    const vals = L.A && L.A[0] ? L.A[0].slice() : [];
    lastNodeColors.raw[k + 1] = vals.slice();
    if (vals.length === 0) {
      lastNodeColors.byLayer[k + 1] = [];
      return;
    }
    if (vals.length === 1) {
      lastNodeColors.byLayer[k + 1] = [clamp01(vals[0])];
      return;
    }
    let min = Infinity,
      max = -Infinity;
    for (const v of vals) {
      if (v < min) min = v;
      if (v > max) max = v;
    }
    const den = max - min || 1;
    lastNodeColors.byLayer[k + 1] = vals.map((v) => (v - min) / den);
  });
}

// ========= Test Inputs UI =========
function renderTestInputs() {
  const container = $("#testInputs");
  if (!container) return;
  const prev = Array.from(container.querySelectorAll("[data-ti]")).map((inp) =>
    Number(inp.value)
  );
  container.innerHTML = "";
  for (let i = 0; i < inputSize; i++) {
    const col = document.createElement("div");
    col.className = "col-6";
    const val = Number.isFinite(prev[i]) ? prev[i] : 0;
    col.innerHTML = `<label class="form-label">x${i + 1}</label>
                   <input type="number" step="any" class="form-control" data-ti="${i}" value="${val}">`;
    container.appendChild(col);
  }
}

// ========= Architettura (render & DnD) =========
function renderArchitecture() {
  const archEl = $("#architecture");
  if (!archEl) return;
  archEl.innerHTML = "";
  if (arch.length === 0) {
    archEl.innerHTML =
      '<div class="text-center small-muted py-4">Trascina qui i layer dalla palette…</div>';
    return;
  }

  arch.forEach((L, idx) => {
    const card = document.createElement("div");
    card.className = "layer-card mb-2";
    card.dataset.id = L.id;
    card.draggable = true;
    const icon =
      L.type === "input"
        ? "bi-box-arrow-in-right text-warning"
        : L.type === "output"
        ? "bi-box-arrow-right text-success"
        : "bi-diagram-3 text-info";
    const name =
      L.type === "input"
        ? "INPUT"
        : L.type === "output"
        ? "OUTPUT"
        : "LAYER NASCOSTO";

    card.innerHTML = `
      <div class="d-flex align-items-center justify-content-between mb-2">
        <div class="d-flex align-items-center gap-2">
          <i class="bi ${icon}"></i>
          <strong>${name}</strong>
          <span class="badge rounded-pill bg-secondary">#${idx + 1}</span>
        </div>
        <button class="btn btn-sm btn-danger remove-layer" type="button" title="Rimuovi">
          <i class="bi bi-x-lg"></i>
        </button>
      </div>
      <div class="row g-2">
        ${
          L.type !== "input"
            ? `<div class="col-6">
            <label class="form-label">Neuroni: <span class="small" id="neuronsVal-${L.id}">${L.neurons}</span></label>
            <input type="range" min="1" max="64" step="1" value="${L.neurons}" class="form-range" data-field="neurons" data-id="${L.id}">
          </div>`
            : ""
        }
        ${
          L.type === "input"
            ? `<div class="col-6">
            <label class="form-label">Dimensione input</label>
            <input type="number" min="1" max="64" value="${L.neurons}" class="form-control" data-field="neurons" data-id="${L.id}">
          </div>`
            : ""
        }
        ${
          L.type !== "input"
            ? `<div class="col-6">
            <label class="form-label">Attivazione</label>
            <select class="form-select" data-field="activation" data-id="${
              L.id
            }">
              ${["relu", "sigmoid", "tanh", "linear"]
                .map(
                  (a) =>
                    `<option value="${a}" ${
                      L.activation === a ? "selected" : ""
                    }>${a}</option>`
                )
                .join("")}
            </select>
          </div>`
            : ""
        }
        ${
          L.type !== "input"
            ? `<div class="col-6 form-check form-switch ms-3">
            <input class="form-check-input" type="checkbox" data-field="bias" data-id="${
              L.id
            }" ${L.bias ? "checked" : ""}>
            <label class="form-check-label">Bias</label>
          </div>`
            : ""
        }
      </div>`;

    // remove
    card.querySelector(".remove-layer").addEventListener("click", (ev) => {
      ev.preventDefault();
      ev.stopPropagation();
      arch = arch.filter((x) => x.id !== L.id);
      renderArchitecture();
      buildNetwork();
      updateJSON();
    });

    // controls
    card.querySelectorAll("[data-field]").forEach((ctrl) => {
      ctrl.addEventListener("input", (e) => {
        const id = e.target.dataset.id;
        const fld = e.target.dataset.field;
        const obj = arch.find((x) => x.id === id);
        if (!obj) return;
        if (fld === "neurons") {
          obj.neurons = Number(e.target.value);
          const sp = $(`#neuronsVal-${id}`);
          if (sp) sp.textContent = obj.neurons;
          if (obj.type === "input") inputSize = obj.neurons;
        }
        if (fld === "activation") obj.activation = e.target.value;
        if (fld === "bias") obj.bias = e.target.checked;
        buildNetwork();
        updateJSON();
      });
    });

    // reorder via drag
    card.addEventListener("dragstart", (ev) => {
      ev.dataTransfer.setData("text/plain", L.id);
      card.classList.add("ghost");
    });
    card.addEventListener("dragend", (_) => card.classList.remove("ghost"));
    card.addEventListener("dragover", (ev) => {
      ev.preventDefault();
      card.classList.add("drop-hint");
    });
    card.addEventListener("dragleave", (_) =>
      card.classList.remove("drop-hint")
    );
    card.addEventListener("drop", (ev) => {
      ev.preventDefault();
      card.classList.remove("drop-hint");
      const id = ev.dataTransfer.getData("text/plain");
      const fromI = arch.findIndex((x) => x.id === id),
        toI = arch.findIndex((x) => x.id === L.id);
      if (fromI < 0 || toI < 0 || fromI === toI) return;
      const [moved] = arch.splice(fromI, 1);
      arch.splice(toI, 0, moved);
      renderArchitecture();
      buildNetwork();
      updateJSON();
    });

    archEl.appendChild(card);
  });
}

function attachArchDnD() {
  const el = $("#architecture");
  if (!el) {
    console.warn("#architecture non trovato");
    return;
  }
  const clone = el.cloneNode(true);
  el.parentNode.replaceChild(clone, el);
  const zone = $("#architecture");

  zone.addEventListener("dragover", (e) => {
    e.preventDefault();
    zone.classList.add("drop-hint");
  });
  zone.addEventListener("dragleave", (_) => zone.classList.remove("drop-hint"));
  zone.addEventListener("drop", (e) => {
    e.preventDefault();
    zone.classList.remove("drop-hint");
    const type = e.dataTransfer.getData("text/plain");
    addLayer(type || "hidden");
    updateJSON();
  });
}

function addLayer(type) {
  const id = crypto.randomUUID();
  if (type === "input") arch.push({ id, type: "input", neurons: inputSize });
  else if (type === "output")
    arch.push({
      id,
      type: "output",
      neurons: outputSize,
      activation: "sigmoid",
      bias: true,
    });
  else
    arch.push({
      id,
      type: "hidden",
      neurons: 4,
      activation: "relu",
      bias: true,
    });
  renderArchitecture();
  buildNetwork();
  updateJSON();
}

// ========= Visualization (SVG) =========
function renderNNVis() {
  const svg = $("#nnVis");
  if (!svg) return;
  const W = 800,
    H = 360;
  const sizes = [inputSize, ...net.layers.map((L) => L.out)];
  const L = sizes.length;
  if (L < 1) {
    svg.innerHTML = "";
    return;
  }

  const xPad = 80,
    yPad = 30;
  const colW = (W - 2 * xPad) / (L - 1 || 1);
  const nodeR = 13;

  const pos = [];
  for (let li = 0; li < L; li++) {
    const n = sizes[li],
      totalH = H - 2 * yPad,
      gap = totalH / (n + 1),
      x = xPad + li * colW;
    for (let ni = 0; ni < n; ni++) {
      const y = yPad + (ni + 1) * gap;
      pos.push({ li, ni, x, y });
    }
  }
  const nodeIndex = (li, ni) =>
    sizes.slice(0, li).reduce((s, v) => s + v, 0) + ni;

  let maxAbs = 1e-6;
  net.layers.forEach((L) => {
    L.W.forEach((r) =>
      r.forEach((v) => {
        const a = Math.abs(v);
        if (a > maxAbs) maxAbs = a;
      })
    );
  });

  const defs = `<defs>
    <filter id="edgeGlow" x="-50%" y="-50%" width="200%" height="200%">
      <feGaussianBlur stdDeviation="1.2" result="blur"/>
      <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
    </filter>
  </defs>`;

  let edges = "";
  net.layers.forEach((Lyr, li) => {
    for (let i = 0; i < Lyr.W.length; i++) {
      for (let j = 0; j < Lyr.W[0].length; j++) {
        const from = pos[nodeIndex(li, i)],
          to = pos[nodeIndex(li + 1, j)];
        const w = Lyr.W[i][j],
          sw = 1 + 5 * (Math.abs(w) / maxAbs),
          stroke = w >= 0 ? "#22c55e" : "#ef4444";
        const cx = (from.x + to.x) / 2,
          cy = from.y + (to.y - from.y) * 0.1;
        edges += `<path d="M ${from.x},${from.y} Q ${cx},${cy} ${to.x},${to.y}"
                  stroke="${stroke}" stroke-width="${sw}" stroke-linecap="round"
                  fill="none" opacity="0.95" filter="url(#edgeGlow)"/>`;
      }
    }
  });

  let nodes = "";
  for (let li = 0; li < L; li++) {
    for (let ni = 0; ni < sizes[li]; ni++) {
      const p = pos[nodeIndex(li, ni)];
      const isInput = li === 0,
        isOutput = li === L - 1;
      const vNorm = lastNodeColors?.byLayer?.[li]?.[ni];
      const vRaw = lastNodeColors?.raw?.[li]?.[ni];
      let fill = "#0b1220";
      if (!isInput && vNorm != null) {
        const t = Math.min(1, Math.max(0, vNorm + 0.15));
        fill = isOutput ? outputColor(t) : hiddenColor(t);
      }
      const label = isInput ? "x" + (ni + 1) : isOutput ? "y" + (ni + 1) : "h";
      const badge =
        !isInput && vRaw != null
          ? `<text x="${p.x}" y="${p.y - (nodeR + 7)}" text-anchor="middle"
                 style="fill:#e5e7eb;font-size:10px;font-weight:600">${vRaw.toFixed(
                   2
                 )}</text>`
          : "";
      nodes += `<g class="nn-node">
                  <circle cx="${p.x}" cy="${p.y}" r="${nodeR}" fill="${fill}"
                          stroke="rgba(255,255,255,0.95)" stroke-width="1.6"/>
                  <text x="${p.x}" y="${p.y + 3}" text-anchor="middle"
                        style="fill:#e5e7eb;font-weight:600">${label}</text>
                  ${badge}
                </g>`;
    }
  }
  svg.innerHTML = defs + `<g>${edges}</g><g>${nodes}</g>`;
}

// ========= Build Network =========
function buildNetwork() {
  let lastSize = inputSize;
  net = new Network();
  arch.forEach((L) => {
    if (L.type === "input") {
      lastSize = L.neurons;
    } else {
      const act = L.activation || "relu";
      const useBias = L.bias !== false;
      net.add(new DenseLayer(lastSize, L.neurons, act, useBias, rng(42)));
      lastSize = L.neurons;
    }
  });
  outputSize = lastSize;
  lastNodeColors = null;
  renderTestInputs();
  renderNNVis();
  updateJSON();
}

// ========= Dataset Preset & CSV =========
function loadPreset(name) {
  if (name === "xor") {
    dataset.X = [
      [0, 0],
      [0, 1],
      [1, 0],
      [1, 1],
    ];
    dataset.y = [[0], [1], [1], [0]];
    inputSize = 2;
    outputSize = 1;
    ensureIOInArch();
    $("#csvInfo").textContent = "Caricato preset XOR (4 esempi)";
  } else if (name === "linsep") {
    const X = [],
      y = [],
      r = rng(7);
    for (let i = 0; i < 200; i++) {
      const a = r(),
        b = r();
      X.push([a, b]);
      y.push([a + b > 1 ? 1 : 0]);
    }
    dataset.X = X;
    dataset.y = y;
    inputSize = 2;
    outputSize = 1;
    ensureIOInArch();
    $("#csvInfo").textContent = "Caricato dataset lineare (200 esempi)";
  }
  renderTestInputs();
}
function ensureIOInArch() {
  let inL = arch.find((l) => l.type === "input");
  if (!inL) {
    inL = { id: crypto.randomUUID(), type: "input", neurons: inputSize };
    arch.unshift(inL);
  } else {
    inL.neurons = inputSize;
  }
  let outL = arch.find((l) => l.type === "output");
  if (!outL) {
    outL = {
      id: crypto.randomUUID(),
      type: "output",
      neurons: outputSize,
      activation: "sigmoid",
      bias: true,
    };
    arch.push(outL);
  } else {
    outL.neurons = outputSize;
    if (!outL.activation) outL.activation = "sigmoid";
  }

  renderArchitecture();
  buildNetwork();
  updateJSON();
}

function handleCSVFile(file) {
  if (!file) return;
  const fr = new FileReader();
  fr.onload = () => {
    const lines = fr.result
      .split(/\r?\n/)
      .map((l) => l.trim())
      .filter(Boolean);
    if (/[a-zA-Z]/.test(lines[0])) lines.shift(); // ignora header semplice
    const data = lines.map((l) => l.split(",").map(Number));
    const cols = data[0].length;
    inputSize = cols - 1;
    outputSize = 1;
    dataset.X = data.map((r) => r.slice(0, cols - 1));
    dataset.y = data.map((r) => [r[cols - 1]]);
    $(
      "#csvInfo"
    ).textContent = `CSV: ${dataset.X.length} esempi, ${inputSize} feature`;
    ensureIOInArch();
    renderTestInputs();
  };
  fr.readAsText(file);
}
function attachCsvDnD() {
  const dz = $("#csvDrop");
  if (!dz) return;
  const clone = dz.cloneNode(true);
  dz.parentNode.replaceChild(clone, dz);
  const el = $("#csvDrop");
  el.addEventListener("dragover", (e) => {
    e.preventDefault();
    el.classList.add("dragover");
  });
  el.addEventListener("dragleave", (_) => el.classList.remove("dragover"));
  el.addEventListener("drop", (e) => {
    e.preventDefault();
    el.classList.remove("dragover");
    handleCSVFile(e.dataTransfer.files?.[0]);
  });
}

// ========= Training / Metrics =========
function getBatches(X, y, batch, rand) {
  const idx = X.map((_, i) => i);
  shuffleInPlace(idx, rand);
  const batches = [];
  for (let i = 0; i < idx.length; i += batch) {
    const slice = idx.slice(i, i + batch);
    batches.push({ X: slice.map((j) => X[j]), y: slice.map((j) => y[j]) });
  }
  return batches;
}
function accuracyBinary(pred, y) {
  let ok = 0;
  for (let i = 0; i < pred.length; i++) {
    const p = pred[i][0] >= 0.5 ? 1 : 0;
    const t = y[i][0];
    ok += p === t ? 1 : 0;
  }
  return ok / pred.length;
}

// Chart.js
function ensureChart() {
  if (chart) return chart;
  const ctx = $("#lossChart");
  chart = new Chart(ctx, {
    type: "line",
    data: {
      labels: [],
      datasets: [
        {
          label: "Loss",
          data: [],
          borderColor: "#3b82f6",
          backgroundColor: "rgba(59,130,246,0.15)",
          borderWidth: 2,
          pointRadius: 0,
          tension: 0.25,
        },
      ],
    },
    options: {
      responsive: true,
      plugins: {
        legend: { labels: { color: "#e5e7eb" } },
        tooltip: {
          backgroundColor: "rgba(15,23,42,0.95)",
          titleColor: "#e5e7eb",
          bodyColor: "#cbd5e1",
          borderColor: "rgba(255,255,255,0.2)",
          borderWidth: 1,
        },
      },
      scales: {
        x: {
          grid: { color: "rgba(255,255,255,0.15)" },
          ticks: { color: "#f1f5f9" },
        },
        y: {
          grid: { color: "rgba(255,255,255,0.15)" },
          ticks: { color: "#f1f5f9" },
        },
      },
    },
  });
  return chart;
}

async function trainLoop() {
  const lr = Number($("#lr").value);
  const epochs = Number($("#epochs").value);
  const batch = Number($("#batch").value);
  const rand = rng(42);

  if (dataset.X.length === 0) {
    alert("Carica o scegli un preset/CSV prima di allenare.");
    return;
  }

  const ch = ensureChart();
  ch.data.labels = [];
  ch.data.datasets[0].data = [];
  ch.update();
  stopFlag = false;
  $("#btnStop").disabled = false;
  $("#btnTrain").disabled = true;

  const X = dataset.X.map((r) => r.slice());
  const y = dataset.y.map((r) => r.slice());

  const probe =
    dataset.X && dataset.X.length > 0
      ? dataset.X[0].slice()
      : Array.from({ length: inputSize }, () => 0);
  const LIVE_VIS_EVERY = 1;

  for (let ep = 1; ep <= epochs; ep++) {
    const batches = getBatches(X, y, batch, rand);
    for (const b of batches) {
      const ypred = net.forward(b.X);
      const [_, dLdy] = mse(ypred, b.y);
      net.backward(dLdy, lr);
    }
    const fullPred = net.forward(X);
    const [L, _g] = mse(fullPred, y);
    const acc = accuracyBinary(fullPred, y);

    $("#lossNow").textContent = L.toFixed(4);
    $("#accNow").textContent = (acc * 100).toFixed(1) + "%";
    ch.data.labels.push(ep);
    ch.data.datasets[0].data.push(L);
    ch.update();

    if (ep % LIVE_VIS_EVERY === 0) {
      computeNodeColorsForInput(probe);
      renderNNVis();
      await new Promise((r) => setTimeout(r, 0));
    }
    if (stopFlag) break;
  }

  $("#btnStop").disabled = true;
  $("#btnTrain").disabled = false;
  updateJSON();
}

// ========= Predict =========
function predictOnce() {
  const vals = $$("#testInputs [data-ti]").map((i) => Number(i.value));
  if (vals.length !== inputSize) {
    alert("Dimensione input non coerente con la rete.");
    return;
  }
  const out = net.forward([vals]);
  computeNodeColorsForInput(vals);
  $("#predictOut").textContent = JSON.stringify(
    out[0].map((v) => Number(v.toFixed(5)))
  );
  renderNNVis();
}

// ========= JSON Export/Import & sync =========
function updateJSON() {
  const j = {
    architecture: arch.map((l) => ({
      type: l.type,
      neurons: l.neurons,
      activation: l.activation || null,
      bias: l.bias ?? null,
    })),
    weights: net.layers.map((l) => ({
      in: l.in,
      out: l.out,
      activation: l.activation,
      useBias: l.useBias,
      W: l.W,
      b: l.b,
    })),
  };
  const ta = $("#jsonArea");
  if (!ta) return;
  ta.value = JSON.stringify(j, null, 2);
}

// Import JSON (file input nascosto #importJSON)
$("#importJSON")?.addEventListener("change", (e) => {
  const input = e.target;
  const file = input.files?.[0];
  if (!file) return;
  const fr = new FileReader();
  fr.onload = () => {
    try {
      const o = JSON.parse(fr.result);

      if (Array.isArray(o.layers)) {
        // formato pesi
        if (!o.layers.length) throw new Error("layers vuoto");
        inputSize = o.layers[0].in ?? inputSize;
        outputSize = o.layers.at(-1).out ?? outputSize;

        arch = [
          { id: crypto.randomUUID(), type: "input", neurons: inputSize },
          ...o.layers.slice(0, -1).map((L) => ({
            id: crypto.randomUUID(),
            type: "hidden",
            neurons: L.out,
            activation: L.activation,
            bias: L.useBias,
          })),
          {
            id: crypto.randomUUID(),
            type: "output",
            neurons: outputSize,
            activation: o.layers.at(-1).activation || "sigmoid",
            bias: o.layers.at(-1).useBias ?? true,
          },
        ];

        net = new Network();
        let last = inputSize;
        net.layers = o.layers.map((L) => {
          const d = new DenseLayer(
            last,
            L.out,
            L.activation,
            L.useBias,
            Math.random
          );
          d.W = L.W;
          d.b = L.b;
          last = L.out;
          return d;
        });

        renderArchitecture();
        renderTestInputs();
        renderNNVis();
        updateJSON();
        alert("✅ Import riuscito (formato pesi).");
      } else if (o.architecture || o.weights) {
        // formato architettura (+ opz pesi)
        if (!Array.isArray(o.architecture))
          throw new Error('manca "architecture"');
        arch = o.architecture.map((L) => ({
          id: crypto.randomUUID(),
          type: L.type,
          neurons: Number(L.neurons),
          activation: L.activation,
          bias: L.bias,
        }));
        const inL = arch.find((l) => l.type === "input");
        const outL = arch.find((l) => l.type === "output");
        if (inL) inputSize = Number(inL.neurons);
        if (outL) outputSize = Number(outL.neurons);

        buildNetwork(); // pesi random coerenti
        if (
          Array.isArray(o.weights) &&
          o.weights.length === net.layers.length
        ) {
          for (let i = 0; i < net.layers.length; i++) {
            if (o.weights[i].W && o.weights[i].b) {
              net.layers[i].W = o.weights[i].W;
              net.layers[i].b = o.weights[i].b;
            }
          }
        }

        renderArchitecture();
        renderTestInputs();
        renderNNVis();
        updateJSON();
        alert(
          "✅ Import riuscito (architettura" +
            (o.weights ? " + pesi" : "") +
            ")."
        );
      } else {
        throw new Error(
          "Formato non riconosciuto. Attesi: {layers:[...]} oppure {architecture:[...], weights?:[...]}"
        );
      }
    } catch (err) {
      console.error(err);
      alert("❌ JSON non valido: " + err.message);
    } finally {
      input.value = "";
    }
  };
  fr.readAsText(file);
});

// CSV file chooser
$("#csvFile")?.addEventListener("change", (e) =>
  handleCSVFile(e.target.files?.[0])
);

// ========= Export helpers =========
function exportArchitecture() {
  const j = {
    architecture: arch.map((l) => ({
      type: l.type,
      neurons: l.neurons,
      activation: l.activation || null,
      bias: l.bias ?? null,
    })),
  };
  const blob = new Blob([JSON.stringify(j, null, 2)], {
    type: "application/json",
  });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "neurobuilder-arch.json";
  a.click();
}

function downloadWeights() {
  const j = {
    layers: net.layers.map((l) => ({
      in: l.in,
      out: l.out,
      activation: l.activation,
      useBias: l.useBias,
      W: l.W,
      b: l.b,
    })),
  };
  const blob = new Blob([JSON.stringify(j, null, 2)], {
    type: "application/json",
  });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "neurobuilder-weights.json";
  a.click();
}

// ========= Popover CSV (dark via customClass + chiusura fuori/ESC) =========
function initCsvInfoSafe() {
  const btn = document.getElementById("csvInfoBtn");
  if (!btn) return;

  btn.setAttribute("type", "button");
  btn.setAttribute("tabindex", "0");
  btn.setAttribute("role", "button");
  btn.setAttribute("aria-label", "Informazioni formato CSV");
  btn.setAttribute("data-bs-toggle", "popover");
  btn.setAttribute("data-bs-theme", "dark"); // opzionale, aiuta con focus/outline

  if (!window.bootstrap || !bootstrap.Popover) {
    console.warn("[CSV Info] Bootstrap.Popover non disponibile");
    return;
  }

  // cleanup eventuale
  const prev = bootstrap.Popover.getInstance(btn);
  if (prev) prev.dispose();
  document.querySelectorAll(".popover").forEach((p) => p.remove());

  const contentHtml = `
    <div>
      <b>• Senza intestazioni</b><br>
      • Separatore: virgola (<code>,</code>)<br>
      • Tutto numerico (niente NaN)<br>
      • <b>Ultima colonna = target</b> (0/1)<br>
      • Esempio:<br>
      <code>0,0,0<br>0,1,1<br>1,0,1<br>1,1,0</code>
    </div>`;

  const pop = new bootstrap.Popover(btn, {
    html: true,
    sanitize: false,
    container: "body",
    placement: "right",
    trigger: "manual",
    title: "Formato CSV richiesto",
    content: contentHtml,
    customClass: "popover-dark", // <<— qui forziamo il tema dark
  });

  const isOpen = () => {
    const id = btn.getAttribute("aria-describedby");
    return !!(id && document.getElementById(id)?.classList.contains("show"));
  };

  btn.addEventListener("click", (e) => {
    e.preventDefault();
    isOpen() ? pop.hide() : pop.show();
    e.stopPropagation();
  });

  document.addEventListener(
    "click",
    (e) => {
      if (!isOpen()) return;
      const id = btn.getAttribute("aria-describedby");
      const tip = id && document.getElementById(id);
      if (btn.contains(e.target) || (tip && tip.contains(e.target))) return;
      pop.hide();
    },
    true
  );

  document.addEventListener(
    "keydown",
    (e) => {
      if (e.key === "Escape") pop.hide();
    },
    true
  );
}

// ========= Binder unico di tutti i bottoni =========
function bindUIControls() {
  // helper per bind sicuro
  const on = (id, ev, handler) => {
    const el = document.getElementById(id);
    if (!el) {
      console.warn("[UI] manca #" + id);
      return null;
    }
    const clone = el.cloneNode(true);
    el.parentNode.replaceChild(clone, el);
    clone.addEventListener(ev, (e) => {
      e.preventDefault();
      handler(e);
    });
    return clone;
  };

  // TRAIN / STOP
  on("btnTrain", "click", () => {
    const t = $("#btnTrain"),
      s = $("#btnStop");
    if (t) t.disabled = true;
    if (s) s.disabled = false;
    trainLoop();
  });
  on("btnStop", "click", () => {
    stopFlag = true;
  });

  // PREDICT
  on("btnPredict", "click", () => {
    const ti = document.querySelectorAll("#testInputs [data-ti]");
    if (ti.length !== inputSize) renderTestInputs();
    predictOnce();
  });

  // QUICK START
  on("btnQuickStart", "click", () => {
    arch = [];
    inputSize = 2;
    outputSize = 1;
    arch.push({ id: crypto.randomUUID(), type: "input", neurons: 2 });
    arch.push({
      id: crypto.randomUUID(),
      type: "hidden",
      neurons: 4,
      activation: "tanh",
      bias: true,
    });
    arch.push({
      id: crypto.randomUUID(),
      type: "output",
      neurons: 1,
      activation: "sigmoid",
      bias: true,
    });
    renderArchitecture();
    buildNetwork();
    loadPreset("xor");
  });

  // PRESET (usa select #presetDataset)
  on("btnLoadPreset", "click", () => {
    const sel = document.getElementById("presetDataset");
    const v = sel?.value || "none";
    if (v !== "none") loadPreset(v);
  });

  // PALETTE
  on("btnAddHidden", "click", () => addLayer("hidden"));
  on("btnClear", "click", () => {
    arch = [];
    renderArchitecture();
    buildNetwork();
    updateJSON();
  });

  // EXPORT / IMPORT JSON
  on("btnExport", "click", () => exportArchitecture());
  on("btnDownloadJSON", "click", () => downloadWeights());
  on("btnCopyJSON", "click", async () => {
    await navigator.clipboard.writeText(
      document.getElementById("jsonArea")?.value || ""
    );
    const b = document.getElementById("btnCopyJSON");
    if (!b) return;
    const txt = b.innerHTML;
    b.innerHTML = '<i class="bi bi-clipboard-check"></i> Copiato!';
    setTimeout(() => (b.innerHTML = txt), 1200);
  });

  // FILE PICKER (se usi bottoni visibili che aprono gli <input type="file">)
  on("btnChooseCSV", "click", () =>
    document.getElementById("csvFile")?.click()
  );
  on("btnImportJSON", "click", () =>
    document.getElementById("importJSON")?.click()
  );

  // slider label live
  const lb = (idIn, idOut) =>
    document
      .getElementById(idIn)
      ?.addEventListener(
        "input",
        (e) => (document.getElementById(idOut).textContent = e.target.value)
      );
  lb("lr", "lrVal");
  lb("epochs", "epochsVal");
  lb("batch", "batchVal");

  // palette DnD
  document.querySelectorAll(".palette-item").forEach((el) => {
    el.addEventListener("dragstart", (ev) =>
      ev.dataTransfer.setData("text/plain", el.dataset.type)
    );
  });
}

// ========= Sanity check opzionale (console) =========
function sanityCheckButtons() {
  const ids = [
    "btnTrain",
    "btnStop",
    "btnPredict",
    "btnQuickStart",
    "btnLoadPreset",
    "btnAddHidden",
    "btnClear",
    "btnExport",
    "btnDownloadJSON",
    "btnCopyJSON",
    "btnImportJSON",
    "csvFile",
    "presetDataset",
    "csvInfoBtn",
  ];
  console.group(
    "%c[NeuroBuilder] Check bottoni",
    "color:#0ea5e9;font-weight:700"
  );
  ids.forEach((id) => {
    const el = document.getElementById(id);
    console[el ? "log" : "warn"](
      `${el ? "✓" : "✗"} ${id} ${el ? "trovato" : "MANCANTE"}`
    );
  });
  console.groupEnd();
}

// ========= Init DOM pronto =========
document.addEventListener("DOMContentLoaded", () => {
  sanityCheckButtons(); // opzionale: puoi rimuoverlo
  attachArchDnD();
  attachCsvDnD();

  // rete di default
  addLayer("input");
  addLayer("hidden");
  addLayer("output");
  buildNetwork();

  // bind bottoni e popover
  bindUIControls();
  initCsvInfoSafe();
});
