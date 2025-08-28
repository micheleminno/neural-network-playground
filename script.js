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

// ========= Neural Net Core =========
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
    if (this.useBias) {
      for (let j = 0; j < this.b[0].length; j++)
        this.b[0][j] -= (lr * dB[0][j]) / r;
    }
    return dX;
  }
}

class Network {
  constructor() {
    this.layers = [];
  }
  add(layer) {
    this.layers.push(layer);
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
let chart;
let stopFlag = false;
let lastNodeColors = null;

// ========= Colors =========
function clamp01(v) {
  return Math.max(0, Math.min(1, v));
}
function lerp(a, b, t) {
  return a + (b - a) * t;
}

/* Hidden: blu scuro → azzurro chiaro (solo luminosità, va benissimo per gli hidden) */
function hiddenColor(v) {
  const t = clamp01(v);
  return `hsl(210 80% ${Math.round(lerp(12, 60, t))}%)`;
}

/* Output: GRADIENTE VERO rosso→giallo→verde in base a t∈[0,1] */
function outputColor(v) {
  const t = clamp01(v);
  // hue 0° (rosso) → 120° (verde); saturazione alta, lightness media
  const hue = Math.round(120 * t);
  return `hsl(${hue} 85% 50%)`;
}

function hiddenColor(v) {
  const t = clamp01(v);
  return `hsl(210 80% ${Math.round(lerp(12, 60, t))}%)`;
}
function computeNodeColorsForInput(xvec) {
  if (!xvec) return;
  net.forward([xvec]);

  lastNodeColors = { byLayer: [], raw: [] }; // raw = attivazioni reali

  net.layers.forEach((L, k) => {
    const vals = L.A && L.A[0] ? L.A[0].slice() : [];
    lastNodeColors.raw[k + 1] = vals.slice();

    if (vals.length === 0) {
      lastNodeColors.byLayer[k + 1] = [];
      return;
    }

    // Caso speciale: layer con 1 solo neurone → usa direttamente il valore (già ~[0,1] se sigmoid)
    if (vals.length === 1) {
      lastNodeColors.byLayer[k + 1] = [Math.max(0, Math.min(1, vals[0]))];
      return;
    }

    // Normalizzazione per-layer su [0,1] (valori multipli)
    let min = Infinity,
      max = -Infinity;
    for (const v of vals) {
      if (v < min) min = v;
      if (v > max) max = v;
    }
    const den = max - min || 1; // qui non capita più 0 perché length>1
    lastNodeColors.byLayer[k + 1] = vals.map((v) => (v - min) / den);
  });
}

// ========= Architettura (drag & drop) =========
const archEl = $("#architecture");

function renderArchitecture() {
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

    const displayIcon =
      L.type === "input"
        ? "bi-box-arrow-in-right text-warning"
        : L.type === "output"
        ? "bi-box-arrow-right text-success"
        : "bi-diagram-3 text-info";
    const displayName =
      L.type === "input"
        ? "INPUT"
        : L.type === "output"
        ? "OUTPUT"
        : "LAYER NASCOSTO";

    card.innerHTML = `
      <div class="d-flex align-items-center justify-content-between mb-2">
        <div class="d-flex align-items-center gap-2">
          <i class="bi ${displayIcon}"></i>
          <strong>${displayName}</strong>
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
          <select class="form-select" data-field="activation" data-id="${L.id}">
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

    card.querySelector(".remove-layer").addEventListener("click", (ev) => {
      ev.preventDefault();
      ev.stopPropagation();
      arch = arch.filter((x) => x.id !== L.id);
      renderArchitecture();
      buildNetwork();
    });

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
          if (obj.type === "input") {
            inputSize = obj.neurons;
          }
        }
        if (fld === "activation") {
          obj.activation = e.target.value;
        }
        if (fld === "bias") {
          obj.bias = e.target.checked;
        }
        buildNetwork();
      });
    });

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
      const fromI = arch.findIndex((x) => x.id === id);
      const toI = arch.findIndex((x) => x.id === L.id);
      if (fromI < 0 || toI < 0 || fromI === toI) return;
      const [moved] = arch.splice(fromI, 1);
      arch.splice(toI, 0, moved);
      renderArchitecture();
      buildNetwork();
    });

    archEl.appendChild(card);
  });
}

function addLayer(type) {
  const id = crypto.randomUUID();
  if (type === "input") {
    arch.push({ id, type: "input", neurons: inputSize });
  } else if (type === "output") {
    arch.push({
      id,
      type: "output",
      neurons: outputSize,
      activation: "sigmoid",
      bias: true,
    });
  } else {
    // 'hidden'
    arch.push({
      id,
      type: "hidden",
      neurons: 4,
      activation: "relu",
      bias: true,
    });
  }
  renderArchitecture();
  buildNetwork();
}

// Drop di nuovi layer nell'area architettura
archEl.addEventListener("dragover", (e) => {
  e.preventDefault();
  archEl.classList.add("drop-hint");
});
archEl.addEventListener("dragleave", (_) =>
  archEl.classList.remove("drop-hint")
);
archEl.addEventListener("drop", (e) => {
  e.preventDefault();
  archEl.classList.remove("drop-hint");
  const type = e.dataTransfer.getData("text/plain"); // palette → 'input' | 'hidden' | 'output'
  addLayer(type || "hidden");
});

// Dragstart sui blocchi palette → passiamo il tipo come text/plain
$$(".palette-item").forEach((el) => {
  el.addEventListener("dragstart", (ev) => {
    ev.dataTransfer.setData("text/plain", el.dataset.type); // 'hidden' al posto di 'dense'
  });
});

// Pulsanti
$("#btnAddHidden").addEventListener("click", () => addLayer("hidden"));
$("#btnClear").addEventListener("click", () => {
  arch = [];
  renderArchitecture();
  buildNetwork();
});

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

  lastNodeColors = null; // reset colori nodi
  renderTestInputs();
  renderNNVis();
  updateJSON();
}

// ========= Visualization (SVG) =========
function renderNNVis() {
  const svg = document.getElementById("nnVis");
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

  // --- Posizioni nodi ---
  const pos = [];
  for (let li = 0; li < L; li++) {
    const n = sizes[li];
    const totalH = H - 2 * yPad;
    const gap = totalH / (n + 1);
    const x = xPad + li * colW;
    for (let ni = 0; ni < n; ni++) {
      const y = yPad + (ni + 1) * gap;
      pos.push({ li, ni, x, y });
    }
  }
  const nodeIndex = (li, ni) =>
    sizes.slice(0, li).reduce((s, v) => s + v, 0) + ni;

  // --- Scala pesi per spessori ---
  let maxAbs = 1e-6;
  net.layers.forEach((L) => {
    L.W.forEach((r) =>
      r.forEach((v) => {
        const a = Math.abs(v);
        if (a > maxAbs) maxAbs = a;
      })
    );
  });

  // --- Defs: glow archi ---
  const defs = `
    <defs>
      <filter id="edgeGlow" x="-50%" y="-50%" width="200%" height="200%">
        <feGaussianBlur stdDeviation="1.2" result="blur"/>
        <feMerge>
          <feMergeNode in="blur"/>
          <feMergeNode in="SourceGraphic"/>
        </feMerge>
      </filter>
    </defs>`;

  // --- Archi (colori ± e spessori dinamici) ---
  let edges = "";
  net.layers.forEach((Lyr, li) => {
    for (let i = 0; i < Lyr.W.length; i++) {
      for (let j = 0; j < Lyr.W[0].length; j++) {
        const from = pos[nodeIndex(li, i)];
        const to = pos[nodeIndex(li + 1, j)];
        const w = Lyr.W[i][j];
        const sw = 1.0 + 5.0 * (Math.abs(w) / maxAbs); // escursione più evidente
        const stroke = w >= 0 ? "#22c55e" : "#ef4444";
        const cx = (from.x + to.x) / 2;
        const cy = from.y + (to.y - from.y) * 0.1; // lieve curvatura

        edges += `<path d="M ${from.x},${from.y} Q ${cx},${cy} ${to.x},${to.y}"
                    stroke="${stroke}"
                    stroke-width="${sw}"
                    stroke-linecap="round"
                    fill="none"
                    opacity="0.95"
                    filter="url(#edgeGlow)"/>`;
      }
    }
  });

  // --- Nodi (colorati in base alle attivazioni normalizzate per layer) ---
  let nodes = "";
  for (let li = 0; li < L; li++) {
    for (let ni = 0; ni < sizes[li]; ni++) {
      const p = pos[nodeIndex(li, ni)];
      const isInput = li === 0;
      const isOutput = li === L - 1;

      // vNorm = per colore (0..1 normalizzato); vRaw = valore reale dell'attivazione
      const vNorm = lastNodeColors?.byLayer?.[li]?.[ni];
      const vRawArr = lastNodeColors?.raw?.[li];
      const vRaw = vRawArr && vRawArr[ni] != null ? vRawArr[ni] : null;

      // Colore: se non input e abbiamo un valore normalizzato, usa palette con un piccolo boost
      let fill = "#0b1220";
      if (!isInput && vNorm != null) {
        const t = Math.min(1, Math.max(0, vNorm + 0.15)); // boost visivo
        fill = isOutput ? outputColor(t) : hiddenColor(t);
      }

      // Etichetta numerica: sugli output mostra SEMPRE il valore reale; sugli hidden se disponibile
      let valueBadge = "";
      if (!isInput && vRaw != null) {
        valueBadge = `<text x="${p.x}" y="${
          p.y - (nodeR + 7)
        }" text-anchor="middle"
                        style="fill:#e5e7eb;font-size:10px;font-weight:600">${vRaw.toFixed(
                          2
                        )}</text>`;
      }

      const label = isInput ? "x" + (ni + 1) : isOutput ? "y" + (ni + 1) : "h";

      nodes += `<g class="nn-node">
                  <circle cx="${p.x}" cy="${p.y}" r="${nodeR}"
                          fill="${fill}"
                          stroke="rgba(255,255,255,0.95)"
                          stroke-width="1.6"/>
                  <text x="${p.x}" y="${p.y + 3}" text-anchor="middle"
                        style="fill:#e5e7eb;font-weight:600">${label}</text>
                  ${valueBadge}
                </g>`;
    }
  }

  svg.innerHTML = defs + `<g>${edges}</g><g>${nodes}</g>`;
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
      y = [];
    const r = rng(7);
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
  if (!arch.some((l) => l.type === "input"))
    arch.unshift({
      id: crypto.randomUUID(),
      type: "input",
      neurons: inputSize,
    });
  if (!arch.some((l) => l.type === "output"))
    arch.push({
      id: crypto.randomUUID(),
      type: "output",
      neurons: outputSize,
      activation: "sigmoid",
      bias: true,
    });
  renderArchitecture();
  buildNetwork();
}
$("#btnLoadPreset").addEventListener("click", () => {
  const v = $("#presetDataset").value;
  if (v !== "none") loadPreset(v);
});
const csvDrop = $("#csvDrop");
csvDrop.addEventListener("dragover", (e) => {
  e.preventDefault();
  csvDrop.classList.add("dragover");
});
csvDrop.addEventListener("dragleave", (_) =>
  csvDrop.classList.remove("dragover")
);
csvDrop.addEventListener("drop", (e) => {
  e.preventDefault();
  csvDrop.classList.remove("dragover");
  handleCSVFile(e.dataTransfer.files[0]);
});
$("#csvFile").addEventListener("change", (e) => {
  handleCSVFile(e.target.files[0]);
});
function handleCSVFile(file) {
  if (!file) return;
  const fr = new FileReader();
  fr.onload = () => {
    const lines = fr.result
      .split(/\r?\n/)
      .map((l) => l.trim())
      .filter(Boolean);
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

// ========= Training =========
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
function updateJSON() {
  // Esporta con label "hidden" per coerenza
  const j = {
    layers: net.layers.map((l) => ({
      type: "hidden",
      in: l.in,
      out: l.out,
      activation: l.activation,
      useBias: l.useBias,
      W: l.W,
      b: l.b,
    })),
  };
  $("#jsonArea").value = JSON.stringify(j, null, 2);
}

// ========= Test inputs UI =========
function renderTestInputs() {
  const c = $("#testInputs");
  if (!c) return;
  c.innerHTML = "";
  for (let i = 0; i < inputSize; i++) {
    const el = document.createElement("div");
    el.className = "col-6";
    el.innerHTML = `<label class="form-label">x${
      i + 1
    }</label><input type="number" step="any" class="form-control" data-ti="${i}" value="0">`;
    c.appendChild(el);
  }
}

// ========= Chart (loss) =========
function ensureChart() {
  if (chart) return chart;
  const ctx = document.getElementById("lossChart");
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
  const LIVE_VIS_EVERY = 1; // ogni epoca

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

    // Live: aggiorna archi (pesi) + nodi (attivazioni sul probe)
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

// ========= Predictions =========
function predictOnce() {
  const vals = $$("#testInputs [data-ti]").map((i) => Number(i.value));
  if (vals.length !== inputSize) {
    alert("Dimensione input non coerente con la rete.");
    return;
  }
  const out = net.forward([vals]);
  computeNodeColorsForInput(vals); // colori hidden + output per l’input
  $("#predictOut").textContent = JSON.stringify(
    out[0].map((v) => Number(v.toFixed(5)))
  );
  renderNNVis(); // aggiorna subito la visualizzazione
}

// ========= Export/Import =========
$("#btnExport").addEventListener("click", () => {
  const blob = new Blob([$("#jsonArea").value], { type: "application/json" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "neurobuilder-arch.json";
  a.click();
});
$("#btnDownloadJSON").addEventListener("click", () => {
  const j = {
    layers: net.layers.map((l) => ({
      type: "hidden",
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
});
$("#importJSON").addEventListener("change", (e) => {
  const file = e.target.files[0];
  if (!file) return;
  const fr = new FileReader();
  fr.onload = () => {
    try {
      const o = JSON.parse(fr.result);
      net = new Network();
      arch = [
        {
          id: crypto.randomUUID(),
          type: "input",
          neurons: o.layers[0]?.in || inputSize,
        },
        ...o.layers.map((L) => ({
          id: crypto.randomUUID(),
          type: "hidden",
          neurons: L.out,
          activation: L.activation,
          bias: L.useBias,
        })),
      ];
      inputSize = o.layers[0]?.in || inputSize;
      outputSize = o.layers.at(-1)?.out || outputSize;
      // ricostruisco i layer con i pesi importati
      let lastSize = inputSize;
      net.layers = o.layers.map((L) => {
        const d = new DenseLayer(
          lastSize,
          L.out,
          L.activation,
          L.useBias,
          Math.random
        );
        d.W = L.W;
        d.b = L.b;
        lastSize = L.out;
        return d;
      });
      renderArchitecture();
      renderTestInputs();
      renderNNVis();
      updateJSON();
    } catch (err) {
      alert("JSON non valido");
    }
  };
  fr.readAsText(file);
});

$("#btnCopyJSON").addEventListener("click", async () => {
  await navigator.clipboard.writeText($("#jsonArea").value);
  const b = $("#btnCopyJSON");
  const txt = b.innerHTML;
  b.innerHTML = '<i class="bi bi-clipboard-check"></i> Copiato!';
  setTimeout(() => (b.innerHTML = txt), 1200);
});

// ========= Controls Bind =========
$("#btnTrain").addEventListener("click", trainLoop);
$("#btnStop").addEventListener("click", () => (stopFlag = true));
$("#btnPredict").addEventListener("click", predictOnce);

$("#lr").addEventListener(
  "input",
  (e) => ($("#lrVal").textContent = e.target.value)
);
$("#epochs").addEventListener(
  "input",
  (e) => ($("#epochsVal").textContent = e.target.value)
);
$("#batch").addEventListener(
  "input",
  (e) => ($("#batchVal").textContent = e.target.value)
);

$("#btnQuickStart").addEventListener("click", () => {
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

// ========= Init =========
addLayer("input");
addLayer("hidden");
addLayer("output");
buildNetwork();
