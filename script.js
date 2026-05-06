// ========= NeuroBuilder - script unico IT/EN =========
// Richiede nell'HTML:
// <body data-lang="it"> oppure <body data-lang="en">
// bottone lingua: #btnLangToggle
// tutti gli id già presenti nel tuo HTML unico.

// ========= Utility =========
const $ = (s) => document.querySelector(s);
const $$ = (s) => Array.from(document.querySelectorAll(s));

function getLang() {
  return document.body?.dataset?.lang === "en" ? "en" : "it";
}

function setLang(lang) {
  const safeLang = lang === "en" ? "en" : "it";

  document.body.dataset.lang = safeLang;
  document.documentElement.lang = safeLang;

  localStorage.setItem("neurobuilder-lang", safeLang);

  applyI18n();

  // forza rebuild UI
  renderArchitecture();
  renderTestInputs();
  renderNNVis();
  updateJSON();
}

function t(key) {
  const lang = getLang();
  return I18N[lang]?.[key] ?? I18N.it[key] ?? key;
}

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

function clamp(x, min, max) {
  return Math.max(min, Math.min(max, x));
}

// ========= Dizionario UI =========
const I18N = {
  it: {
    input: "Input",
    output: "Output",
    hiddenLayer: "Layer nascosto",
    neurons: "Neuroni",
    inputSize: "Dimensione input",
    activation: "Attivazione",
    bias: "Bias",
    langButton: "EN",
    emptyArchitecture: "Trascina qui i layer dalla palette…",
    remove: "Rimuovi",
    csvNoFile: "Nessun file selezionato.",
    csvEmpty: "Il file sembra vuoto.",
    csvNeedCols: "Servono almeno 2 colonne (feature + target).",
    csvNonNumeric: "Valori non numerici alla riga",
    csvLoaded: "caricato",
    csvExamples: "esempi",
    csvFeatures: "feature",
    csvDelimiter: "delimitatore",
    csvNoDataset: "Nessun dataset caricato",
    csvReadError: "Impossibile leggere il file CSV.",
    csvParseError: "Errore CSV: ",
    presetXorLoaded: "Caricato preset XOR (4 esempi)",
    presetLinearLoaded: "Caricato dataset lineare (200 esempi)",
    trainNoDataset: "Carica o scegli un preset/CSV prima di allenare.",
    inputMismatch: "Dimensione input non coerente con la rete.",
    importWeightsOk: "✅ Import riuscito (formato pesi).",
    importArchOk: "✅ Import riuscito (architettura",
    importWeightsSuffix: " + pesi",
    importClose: ").",
    jsonInvalid: "❌ JSON non valido: ",
    jsonUnknown:
      "Formato non riconosciuto. Attesi: {layers:[...]} oppure {architecture:[...], weights?:[...]}",
    layersEmpty: "layers vuoto",
    architectureMissing: 'manca "architecture"',
    copied: '<i class="bi bi-clipboard-check"></i> Copiato!',
    csvPopoverTitle: "Formato CSV richiesto",
    csvPopoverHtml: `
      <div>
        <b>• Senza intestazioni</b><br>
        • Separatore: virgola (<code>,</code>)<br>
        • Tutto numerico (niente NaN)<br>
        • <b>Ultima colonna = target</b> (0/1)<br>
        • Esempio:<br>
        <code>0,0,0<br>0,1,1<br>1,0,1<br>1,1,0</code>
      </div>`,
  },
  en: {
    input: "Input",
    output: "Output",
    hiddenLayer: "Hidden layer",
    neurons: "Neurons",
    inputSize: "Input size",
    activation: "Activation",
    bias: "Bias",
    langButton: "IT",
    emptyArchitecture: "Drag layers here from the palette…",
    remove: "Remove",
    csvNoFile: "No file selected.",
    csvEmpty: "The file seems empty.",
    csvNeedCols: "At least 2 columns are required (features + target).",
    csvNonNumeric: "Non-numeric values at row",
    csvLoaded: "loaded",
    csvExamples: "examples",
    csvFeatures: "features",
    csvDelimiter: "delimiter",
    csvNoDataset: "No dataset loaded",
    csvReadError: "Can't read CSV file.",
    csvParseError: "CSV error: ",
    presetXorLoaded: "XOR preset loaded (4 examples)",
    presetLinearLoaded: "Linear dataset loaded (200 examples)",
    trainNoDataset: "Load or choose a preset/CSV before training.",
    inputMismatch: "Input size doesn't match the network.",
    importWeightsOk: "✅ Import OK (weight format).",
    importArchOk: "✅ Import OK (Architecture",
    importWeightsSuffix: " + weights",
    importClose: ").",
    jsonInvalid: "❌ JSON not valid: ",
    jsonUnknown:
      "Format unknown. Expected: {layers:[...]} or {architecture:[...], weights?:[...]}",
    layersEmpty: "layers empty",
    architectureMissing: '"Architecture" missing',
    copied: '<i class="bi bi-clipboard-check"></i> Copied!',
    csvPopoverTitle: "Required CSV format",
    csvPopoverHtml: `
      <div>
        <b>• No headers</b><br>
        • Separator: comma (<code>,</code>)<br>
        • All numeric values (no NaN)<br>
        • <b>Last column = target</b> (0/1)<br>
        • Example:<br>
        <code>0,0,0<br>0,1,1<br>1,0,1<br>1,1,0</code>
      </div>`,
  },
};

function applyI18n() {
  const lang = getLang();

  $$("[data-i18n]").forEach((el) => {
    const key = el.dataset.i18n;

    if (I18N_HTML[lang]?.[key] !== undefined) {

      if (el.tagName === "OPTION") {
        el.textContent = I18N_HTML[lang][key];
      } else {
        el.innerHTML = I18N_HTML[lang][key];
      }

    }
  });

  $$("[data-i18n-placeholder]").forEach((el) => {
    const key = el.dataset.i18nPlaceholder;

    if (I18N_HTML[lang]?.[key] !== undefined) {
      el.placeholder = I18N_HTML[lang][key];
    }
  });

  $$("[data-i18n-aria]").forEach((el) => {
    const key = el.dataset.i18nAria;

    if (I18N_HTML[lang]?.[key] !== undefined) {
      el.setAttribute("aria-label", I18N_HTML[lang][key]);
    }
  });

  const btn = document.getElementById("btnLangToggle");
  if (btn) {
    btn.textContent = t("langButton");
  }

  const csvInfo = document.getElementById("csvInfo");

  if (csvInfo && (!dataset.X.length || csvInfo.dataset.auto === "empty")) {
    csvInfo.textContent = t("csvNoDataset");
    csvInfo.dataset.auto = "empty";
  }

  initCsvInfoSafe();
}

const I18N_HTML = {
  it: {
    palette: "Palette",
    dragHere: "Trascina i blocchi qui sotto 👉",

    input: "Input",
    output: "Output",
    hiddenLayer: "Layer nascosto",
    neurons: "Neuroni",
    inputSize: "Dimensione input",
    activation: "Attivazione",
    bias: "Bias",

    clear: "Pulisci",
    architecture: "Architettura",
    training: "Training",
    learningRate: "Learning rate",
    epochs: "Epoche",
    batch: "Batch",
    train: "Allena",
    stop: "Stop",
    quickStart: "Quick Start (XOR)",
    loss: "Loss",
    accuracy: "Accuratezza",
    network: "Visualizzazione rete",
    prediction: "Test predizione",
    predict: "Predici",
    outputLabel: "Output",
    dataset: "Dataset",
    load: "Carica",
    csvDropText: "Trascina CSV qui",
    noDataset: "Nessun dataset caricato",
    json: "JSON",
    export: "Export",
    download: "Download",
    copy: "Copy",

    linearPreset: "Lineare"
  },

  en: {
    palette: "Palette",
    dragHere: "Drag the blocks below 👉",

    input: "Input",
    output: "Output",
    hiddenLayer: "Hidden layer",
    neurons: "Neurons",
    inputSize: "Input size",
    activation: "Activation",
    bias: "Bias",

    clear: "Clear",
    architecture: "Architecture",
    training: "Training",
    learningRate: "Learning rate",
    epochs: "Epochs",
    batch: "Batch",
    train: "Train",
    stop: "Stop",
    quickStart: "Quick Start (XOR)",
    loss: "Loss",
    accuracy: "Accuracy",
    network: "Network visualization",
    prediction: "Prediction Test",
    predict: "Predict",
    outputLabel: "Output",
    dataset: "Dataset",
    load: "Load",
    csvDropText: "Drop CSV here",
    noDataset: "No dataset loaded",
    json: "JSON",
    export: "Export",
    download: "Download",
    copy: "Copy",

    linearPreset: "Linear"
  }
};

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
  for (let i = 0; i < r; i++) {
    for (let j = 0; j < c; j++) {
      let u = 0;
      let v = 0;
      while (u === 0) u = rand();
      while (v === 0) v = rand();
      const z = Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
      const scale = Math.sqrt(2 / r); // He initialization, buona per ReLU e accettabile didatticamente
      m[i][j] = z * scale;
    }
  }
  return m;
}

function dot(a, b) {
  const r = a.length;
  const c = b[0].length;
  const n = b.length;
  const m = zeros(r, c);
  for (let i = 0; i < r; i++) {
    for (let j = 0; j < c; j++) {
      let s = 0;
      for (let k = 0; k < n; k++) s += a[i][k] * b[k][j];
      m[i][j] = s;
    }
  }
  return m;
}

function addBias(a, b) {
  const r = a.length;
  const c = a[0].length;
  const m = zeros(r, c);
  for (let i = 0; i < r; i++) {
    for (let j = 0; j < c; j++) m[i][j] = a[i][j] + b[0][j];
  }
  return m;
}

function applyActivation(a, act) {
  const r = a.length;
  const c = a[0].length;
  const m = zeros(r, c);
  for (let i = 0; i < r; i++) {
    for (let j = 0; j < c; j++) m[i][j] = act.f(a[i][j]);
  }
  return m;
}

function hadamard(a, b) {
  const r = a.length;
  const c = a[0].length;
  const m = zeros(r, c);
  for (let i = 0; i < r; i++) {
    for (let j = 0; j < c; j++) m[i][j] = a[i][j] * b[i][j];
  }
  return m;
}

function transpose(a) {
  const r = a.length;
  const c = a[0].length;
  const m = zeros(c, r);
  for (let i = 0; i < r; i++) {
    for (let j = 0; j < c; j++) m[j][i] = a[i][j];
  }
  return m;
}

function bce(pred, target) {
  const r = pred.length;
  const c = pred[0].length;
  let loss = 0;
  const grad = zeros(r, c);
  const eps = 1e-7;

  for (let i = 0; i < r; i++) {
    for (let j = 0; j < c; j++) {
      const p = clamp(pred[i][j], eps, 1 - eps);
      const targetValue = target[i][j];
      loss += -(
        targetValue * Math.log(p) +
        (1 - targetValue) * Math.log(1 - p)
      );
      grad[i][j] = p - targetValue; // da usare con output sigmoid: dZ = p - t
    }
  }

  return [loss / r, grad];
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

  backward(dA, lr, isLastLayer = false) {
    const act = Activations[this.activation];
    const r = this.A.length;
    const c = this.A[0].length;

    const isSigmoidOutput = this.activation === "sigmoid" && isLastLayer;

    const dAct = zeros(r, c);
    for (let i = 0; i < r; i++) {
      for (let j = 0; j < c; j++) dAct[i][j] = act.df(this.A[i][j]);
    }

    const dZ = isSigmoidOutput ? dA : hadamard(dA, dAct);

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

    const CLIP = 1.0;

    for (let i = 0; i < this.W.length; i++) {
      for (let j = 0; j < this.W[0].length; j++) {
        let g = dW[i][j] / r;
        g = clamp(g, -CLIP, CLIP);
        this.W[i][j] -= lr * g;
      }
    }

    if (this.useBias) {
      for (let j = 0; j < this.b[0].length; j++) {
        let g = dB[0][j] / r;
        g = clamp(g, -CLIP, CLIP);
        this.b[0][j] -= lr * g;
      }
    }

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
    let grad = dA;
    for (let i = this.layers.length - 1; i >= 0; i--) {
      const isLastLayer = i === this.layers.length - 1;
      grad = this.layers[i].backward(grad, lr, isLastLayer);
    }
  }
}

// ========= State =========
let net = new Network();
let arch = []; // [{id,type:'input'|'hidden'|'output', neurons, activation, bias}]
let inputSize = 2;
let outputSize = 1;
let dataset = { X: [], y: [] };
let chart;
let stopFlag = false;
let lastNodeColors = null; // {byLayer:[], raw:[]}
let rebuildSeedCounter = 1;

// ========= Colors & Node Coloring =========
const clamp01 = (v) => Math.max(0, Math.min(1, v));
const lerp = (a, b, t) => a + (b - a) * t;
const hiddenColor = (v) =>
  `hsl(210, 80%, ${Math.round(lerp(12, 60, clamp01(v)))}%)`;
const outputColor = (v) => `hsl(${Math.round(120 * clamp01(v))}, 85%, 50%)`;

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

    let min = Infinity;
    let max = -Infinity;
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

// ========= Architettura =========
function renderArchitecture() {
  const archEl = $("#architecture");
  if (!archEl) return;

  archEl.innerHTML = "";

  if (arch.length === 0) {
    archEl.innerHTML = `<div class="text-center small-muted py-4">${t(
      "emptyArchitecture"
    )}</div>`;
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

    const layerNames = {
      input: t("input"),
      output: t("output"),
      hidden: t("hiddenLayer"),
    };

    const name = layerNames[L.type] || L.type;

    card.innerHTML = `
      <div class="d-flex align-items-center justify-content-between mb-2">
        <div class="d-flex align-items-center gap-2">
          <i class="bi ${icon}"></i>
          <strong>${name}</strong>
          <span class="badge rounded-pill bg-secondary">#${idx + 1}</span>
        </div>
        <button class="btn btn-sm btn-danger remove-layer" type="button" title="${t(
          "remove"
        )}">
          <i class="bi bi-x-lg"></i>
        </button>
      </div>
      <div class="row g-2">
        ${
          L.type !== "input"
            ? `<div class="col-6">
                <label class="form-label">${t(
                  "neurons"
                )}: <span class="small" id="neuronsVal-${L.id}">${
                L.neurons
              }</span></label>
                <input type="range" min="1" max="64" step="1" value="${
                  L.neurons
                }" class="form-range" data-field="neurons" data-id="${L.id}">
              </div>`
            : ""
        }
        ${
          L.type === "input"
            ? `<div class="col-6">
                <label class="form-label">${t("inputSize")}</label>
                <input type="number" min="1" max="64" value="${
                  L.neurons
                }" class="form-control" data-field="neurons" data-id="${L.id}">
              </div>`
            : ""
        }
        ${
          L.type !== "input"
            ? `<div class="col-6">
                <label class="form-label">${t("activation")}</label>
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
                <label class="form-check-label">${t("bias")}</label>
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
      updateJSON();
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
          if (obj.type === "input") inputSize = obj.neurons;
        }
        if (fld === "activation") obj.activation = e.target.value;
        if (fld === "bias") obj.bias = e.target.checked;

        buildNetwork();
        updateJSON();
      });
    });

    card.addEventListener("dragstart", (ev) => {
      ev.dataTransfer.setData("text/plain", L.id);
      card.classList.add("ghost");
    });

    card.addEventListener("dragend", () => card.classList.remove("ghost"));

    card.addEventListener("dragover", (ev) => {
      ev.preventDefault();
      card.classList.add("drop-hint");
    });

    card.addEventListener("dragleave", () =>
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
      updateJSON();
    });

    archEl.appendChild(card);
  });
}

function attachArchDnD() {
  const el = document.getElementById("architecture");
  if (!el) {
    console.warn("#architecture not found");
    return;
  }

  const clone = el.cloneNode(true);
  el.parentNode.replaceChild(clone, el);
  const zone = document.getElementById("architecture");

  zone.addEventListener("dragover", (e) => {
    e.preventDefault();
    zone.classList.add("drop-hint");
  });

  zone.addEventListener("dragleave", () => zone.classList.remove("drop-hint"));

  zone.addEventListener("drop", (e) => {
    e.preventDefault();
    zone.classList.remove("drop-hint");

    const payload = e.dataTransfer.getData("text/plain");
    if (payload === "input" || payload === "hidden" || payload === "output") {
      addLayer(payload);
      updateJSON();
    }
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
  updateJSON();
}

// ========= Visualization =========
function renderNNVis() {
  const svg = $("#nnVis");
  if (!svg) return;

  const W = 800;
  const H = 360;
  const sizes = [inputSize, ...net.layers.map((L) => L.out)];
  const layerCount = sizes.length;

  if (layerCount < 1) {
    svg.innerHTML = "";
    return;
  }

  const xPad = 80;
  const yPad = 30;
  const colW = (W - 2 * xPad) / (layerCount - 1 || 1);
  const nodeR = 13;

  const pos = [];
  for (let li = 0; li < layerCount; li++) {
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

  let maxAbs = 1e-6;
  net.layers.forEach((L) => {
    L.W.forEach((row) => {
      row.forEach((v) => {
        const a = Math.abs(v);
        if (a > maxAbs) maxAbs = a;
      });
    });
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
        const from = pos[nodeIndex(li, i)];
        const to = pos[nodeIndex(li + 1, j)];
        const w = Lyr.W[i][j];
        const sw = 1 + 5 * (Math.abs(w) / maxAbs);
        const stroke = w >= 0 ? "#22c55e" : "#ef4444";
        const cx = (from.x + to.x) / 2;
        const cy = from.y + (to.y - from.y) * 0.1;

        edges += `<path d="M ${from.x},${from.y} Q ${cx},${cy} ${to.x},${to.y}"
          stroke="${stroke}" stroke-width="${sw}" stroke-linecap="round"
          fill="none" opacity="0.95" filter="url(#edgeGlow)"/>`;
      }
    }
  });

  let nodes = "";
  for (let li = 0; li < layerCount; li++) {
    for (let ni = 0; ni < sizes[li]; ni++) {
      const p = pos[nodeIndex(li, ni)];
      const isInput = li === 0;
      const isOutput = li === layerCount - 1;
      const vNorm = lastNodeColors?.byLayer?.[li]?.[ni];
      const vRaw = lastNodeColors?.raw?.[li]?.[ni];

      let fill = "#0b1220";
      if (!isInput && vNorm != null) {
        const tt = clamp01(vNorm + 0.15);
        fill = isOutput ? outputColor(tt) : hiddenColor(tt);
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
        <circle cx="${p.x}" cy="${p.y}" r="${nodeR}"
          style="fill:${fill} !important; stroke:rgba(255,255,255,0.95); stroke-width:1.6"/>
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

  // seed diverso a ogni rebuild, ma stabile dentro il rebuild
  const rand = rng(Date.now() + rebuildSeedCounter++);

  arch.forEach((L) => {
    if (L.type === "input") {
      lastSize = L.neurons;
    } else {
      const act = L.activation || "relu";
      const useBias = L.bias !== false;
      net.add(new DenseLayer(lastSize, L.neurons, act, useBias, rand));
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
    const csvInfo = $("#csvInfo");
    if (csvInfo) {
      csvInfo.textContent = t("presetXorLoaded");
      csvInfo.dataset.auto = "loaded";
    }
  } else if (name === "linsep") {
    const X = [];
    const y = [];
    const r = rng(7);
    for (let i = 0; i < 200; i++) {
      const a = r();
      const b = r();
      X.push([a, b]);
      y.push([a + b > 1 ? 1 : 0]);
    }
    dataset.X = X;
    dataset.y = y;
    inputSize = 2;
    outputSize = 1;
    ensureIOInArch();
    const csvInfo = $("#csvInfo");
    if (csvInfo) {
      csvInfo.textContent = t("presetLinearLoaded");
      csvInfo.dataset.auto = "loaded";
    }
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
  if (!file) {
    alert(t("csvNoFile"));
    return;
  }

  console.log("[CSV] reading file:", file.name, file.type, file.size, "bytes");

  const infoEl = document.getElementById("csvInfo");
  const setInfo = (msg, mode = "loaded") => {
    if (infoEl) {
      infoEl.textContent = msg;
      infoEl.dataset.auto = mode;
    }
  };

  const sniffDelimiter = (text) => {
    const counts = {
      ",": (text.match(/,/g) || []).length,
      ";": (text.match(/;/g) || []).length,
      "\t": (text.match(/\t/g) || []).length,
    };
    return Object.entries(counts).sort((a, b) => b[1] - a[1])[0][0] || ",";
  };

  const toNumber = (token) => {
    const value = token.trim().replace(",", ".");
    const n = Number(value);
    return Number.isFinite(n) ? n : NaN;
  };

  const parse = (text) => {
    let lines = text
      .split(/\r?\n/)
      .map((line) => line.trim())
      .filter(Boolean);

    if (lines.length === 0) throw new Error(t("csvEmpty"));

    const delim = sniffDelimiter(lines.slice(0, 50).join("\n"));
    if (/[a-zA-Z]/.test(lines[0])) lines.shift();

    const rows = lines.map((line) => line.split(delim));
    const cols = rows[0].length;
    if (cols < 2) throw new Error(t("csvNeedCols"));

    const data = rows.map((row, ri) => {
      const nums = row.map(toNumber);
      if (nums.some((x) => Number.isNaN(x))) {
        throw new Error(`${t("csvNonNumeric")} ${ri + 1}.`);
      }
      return nums;
    });

    const inputSizeNew = cols - 1;
    const X = data.map((row) => row.slice(0, inputSizeNew));
    const y = data.map((row) => [row[cols - 1]]);
    return { X, y, inputSizeNew, delimiter: delim };
  };

  const fr = new FileReader();

  fr.onload = () => {
    try {
      const text = fr.result;
      const { X, y, inputSizeNew, delimiter } = parse(text);

      dataset.X = X;
      dataset.y = y;
      inputSize = inputSizeNew;
      outputSize = 1;

      ensureIOInArch();
      renderTestInputs();

      setInfo(
        `CSV "${file.name}" ${t("csvLoaded")}: ${X.length} ${t(
          "csvExamples"
        )}, ${inputSize} ${t("csvFeatures")} (${t("csvDelimiter")} "${
          delimiter === "\t" ? "TAB" : delimiter
        }")`
      );

      console.log("[CSV] OK. First rows:", X.slice(0, 3), y.slice(0, 3));
    } catch (err) {
      console.error("[CSV] Parsing error:", err);
      setInfo(t("csvNoDataset"), "empty");
      alert("❌ " + t("csvParseError") + err.message);
    }
  };

  fr.onerror = () => {
    console.error("[CSV] FileReader error:", fr.error);
    setInfo(t("csvNoDataset"), "empty");
    alert("❌ " + t("csvReadError"));
  };

  fr.readAsText(file);
}

function wireCsvInputs() {
  const fi = document.getElementById("csvFile");
  if (fi) {
    if (!fi._wiredCsv) {
      const clone = fi.cloneNode(true);
      fi.parentNode.replaceChild(clone, fi);
      clone._wiredCsv = true;
      clone.addEventListener("change", (e) => {
        const f = e.target.files && e.target.files[0];
        console.log("[CSV] change →", f?.name || "(no file)");
        handleCSVFile(f);
      });
    }
  } else {
    console.warn("[CSV] #csvFile not found");
  }

  const dz = document.getElementById("csvDrop");
  if (dz && !dz._wiredDrop) {
    dz._wiredDrop = true;
    dz.addEventListener("dragover", (e) => {
      e.preventDefault();
      dz.classList.add("dragover");
    });
    dz.addEventListener("dragleave", () => dz.classList.remove("dragover"));
    dz.addEventListener("drop", (e) => {
      e.preventDefault();
      dz.classList.remove("dragover");
      const file = e.dataTransfer?.files?.[0];
      handleCSVFile(file);
    });
  }
}

function observeCsvInputs() {
  const obs = new MutationObserver(() => {
    const fi = document.getElementById("csvFile");
    if (fi && !fi._wiredCsv) wireCsvInputs();
  });
  obs.observe(document.documentElement, { childList: true, subtree: true });
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

  el.addEventListener("dragleave", () => el.classList.remove("dragover"));

  el.addEventListener("drop", (e) => {
    e.preventDefault();
    el.classList.remove("dragover");
    handleCSVFile(e.dataTransfer.files?.[0]);
  });
}

// ========= Training / Metrics =========
function getBatches(X, y, batch, rand) {
  const safeBatch = Math.max(1, Math.min(Number(batch) || 1, X.length));
  const idx = X.map((_, i) => i);
  shuffleInPlace(idx, rand);

  const batches = [];
  for (let i = 0; i < idx.length; i += safeBatch) {
    const slice = idx.slice(i, i + safeBatch);
    batches.push({ X: slice.map((j) => X[j]), y: slice.map((j) => y[j]) });
  }
  return batches;
}

function accuracyBinary(pred, y) {
  if (!pred.length) return 0;
  let ok = 0;
  for (let i = 0; i < pred.length; i++) {
    const p = pred[i][0] >= 0.5 ? 1 : 0;
    const targetValue = y[i][0];
    ok += p === targetValue ? 1 : 0;
  }
  return ok / pred.length;
}

function ensureChart() {
  if (chart) return chart;

  const ctx = $("#lossChart");
  if (!ctx) return null;

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
  const lr = Number(document.getElementById("lr")?.value ?? 0.1);
  const epochs = Number(document.getElementById("epochs")?.value ?? 50);
  const rand = rng(42);

  if (dataset.X.length === 0) {
    alert(t("trainNoDataset"));
    return;
  }

  const ch = ensureChart();
  if (ch) {
    ch.data.labels = [];
    ch.data.datasets[0].data = [];
    ch.update();
  }

  stopFlag = false;
  const btnStop = document.getElementById("btnStop");
  const btnTrain = document.getElementById("btnTrain");
  if (btnStop) btnStop.disabled = false;
  if (btnTrain) btnTrain.disabled = true;

  const X = dataset.X.map((row) => row.slice());
  const y = dataset.y.map((row) => row.slice());

  let step = 0;
  const VIS_EVERY_STEPS = 2;

  for (let ep = 1; ep <= epochs; ep++) {
    const lrNow = lr * 0.995 ** ep;
    const batchSize = Math.min(
      Number(document.getElementById("batch")?.value ?? 4),
      dataset.X.length
    );
    const batches = getBatches(X, y, batchSize, rand);

    for (const b of batches) {
      const ypred = net.forward(b.X);
      const [, dLdy] = bce(ypred, b.y);
      net.backward(dLdy, lrNow);

      step++;
      if (step % VIS_EVERY_STEPS === 0 && b.X.length > 0) {
        computeNodeColorsForInput(b.X[0]);
        renderNNVis();
        await new Promise((resolve) => setTimeout(resolve, 0));
      }
    }

    const fullPred = net.forward(X);
    const [L] = bce(fullPred, y);
    const acc = accuracyBinary(fullPred, y);

    console.log(
      `Epoch ${ep} → Loss: ${L.toFixed(4)} | Acc: ${(acc * 100).toFixed(
        1
      )}% | LR: ${lrNow.toFixed(5)}`
    );

    const lossNow = document.getElementById("lossNow");
    const accNow = document.getElementById("accNow");
    if (lossNow) lossNow.textContent = L.toFixed(4);
    if (accNow) accNow.textContent = (acc * 100).toFixed(1) + "%";

    if (ch) {
      ch.data.labels.push(ep);
      ch.data.datasets[0].data.push(L);
      ch.update();
    }

    if (stopFlag) break;
  }

  if (btnStop) btnStop.disabled = true;
  if (btnTrain) btnTrain.disabled = false;
  updateJSON();
}

// ========= Predict =========
function predictOnce() {
  const vals = $$("#testInputs [data-ti]").map((i) => Number(i.value));
  if (vals.length !== inputSize) {
    alert(t("inputMismatch"));
    return;
  }

  const out = net.forward([vals]);
  computeNodeColorsForInput(vals);

  const predictOut = $("#predictOut");
  if (predictOut) {
    predictOut.textContent = JSON.stringify(
      out[0].map((v) => Number(v.toFixed(5)))
    );
  }

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

function handleJSONImportFile(file, inputEl) {
  if (!file) return;

  const fr = new FileReader();
  fr.onload = () => {
    try {
      const o = JSON.parse(fr.result);

      if (Array.isArray(o.layers)) {
        if (!o.layers.length) throw new Error(t("layersEmpty"));

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
        alert(t("importWeightsOk"));
      } else if (o.architecture || o.weights) {
        if (!Array.isArray(o.architecture))
          throw new Error(t("architectureMissing"));

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

        buildNetwork();

        if (
          Array.isArray(o.weights) &&
          o.weights.length === net.layers.length
        ) {
          for (let i = 0; i < net.layers.length; i++) {
            if (o.weights[i].W) net.layers[i].W = o.weights[i].W;
            if (o.weights[i].b !== undefined) net.layers[i].b = o.weights[i].b;
          }
        }

        renderArchitecture();
        renderTestInputs();
        renderNNVis();
        updateJSON();
        alert(
          t("importArchOk") +
            (o.weights ? t("importWeightsSuffix") : "") +
            t("importClose")
        );
      } else {
        throw new Error(t("jsonUnknown"));
      }
    } catch (err) {
      console.error(err);
      alert(t("jsonInvalid") + err.message);
    } finally {
      if (inputEl) inputEl.value = "";
    }
  };

  fr.readAsText(file);
}

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
  URL.revokeObjectURL(a.href);
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
  URL.revokeObjectURL(a.href);
}

// ========= Popover CSV =========
function initCsvInfoSafe() {
  const btn = document.getElementById("csvInfoBtn");
  if (!btn) return;

  btn.setAttribute("type", "button");
  btn.setAttribute("tabindex", "0");
  btn.setAttribute("role", "button");
  btn.setAttribute(
    "aria-label",
    getLang() === "it" ? "Informazioni formato CSV" : "CSV format info"
  );
  btn.setAttribute("data-bs-toggle", "popover");
  btn.setAttribute("data-bs-theme", "dark");

  if (!window.bootstrap || !bootstrap.Popover) {
    console.warn("[CSV Info] Bootstrap.Popover not available");
    return;
  }

  const prev = bootstrap.Popover.getInstance(btn);
  if (prev) prev.dispose();
  document.querySelectorAll(".popover").forEach((p) => p.remove());

  const pop = new bootstrap.Popover(btn, {
    html: true,
    sanitize: false,
    container: "body",
    placement: "right",
    trigger: "manual",
    title: t("csvPopoverTitle"),
    content: t("csvPopoverHtml"),
    customClass: "popover-dark",
  });

  const isOpen = () => {
    const id = btn.getAttribute("aria-describedby");
    return !!(id && document.getElementById(id)?.classList.contains("show"));
  };

  btn.onclick = (e) => {
    e.preventDefault();
    isOpen() ? pop.hide() : pop.show();
    e.stopPropagation();
  };
}

function attachPopoverGlobalClosers() {
  document.addEventListener(
    "click",
    (e) => {
      const btn = document.getElementById("csvInfoBtn");
      if (!btn || !window.bootstrap || !bootstrap.Popover) return;

      const pop = bootstrap.Popover.getInstance(btn);
      if (!pop) return;

      const id = btn.getAttribute("aria-describedby");
      const tip = id && document.getElementById(id);
      if (!tip || !tip.classList.contains("show")) return;
      if (btn.contains(e.target) || tip.contains(e.target)) return;
      pop.hide();
    },
    true
  );

  document.addEventListener(
    "keydown",
    (e) => {
      if (e.key !== "Escape") return;
      const btn = document.getElementById("csvInfoBtn");
      if (!btn || !window.bootstrap || !bootstrap.Popover) return;
      const pop = bootstrap.Popover.getInstance(btn);
      if (pop) pop.hide();
    },
    true
  );
}

// ========= Binding =========
function bindUIControls() {
  const on = (id, ev, handler) => {
    const el = document.getElementById(id);
    if (!el) {
      console.warn("[UI] Missing #" + id);
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

  on("btnTrain", "click", () => {
    const tbtn = $("#btnTrain");
    const sbtn = $("#btnStop");
    if (tbtn) tbtn.disabled = true;
    if (sbtn) sbtn.disabled = false;
    trainLoop();
  });

  on("btnStop", "click", () => {
    stopFlag = true;
  });

  on("btnPredict", "click", () => {
    const ti = document.querySelectorAll("#testInputs [data-ti]");
    if (ti.length !== inputSize) renderTestInputs();
    predictOnce();
  });

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

  on("btnLoadPreset", "click", () => {
    const sel = document.getElementById("presetDataset");
    const v = sel?.value || "none";
    if (v !== "none") loadPreset(v);
  });

  on("btnAddHidden", "click", () => addLayer("hidden"));

  on("btnClear", "click", () => {
    arch = [];
    renderArchitecture();
    buildNetwork();
    updateJSON();
  });

  on("btnExport", "click", () => exportArchitecture());
  on("btnDownloadJSON", "click", () => downloadWeights());

  on("btnCopyJSON", "click", async () => {
    await navigator.clipboard.writeText(
      document.getElementById("jsonArea")?.value || ""
    );
    const b = document.getElementById("btnCopyJSON");
    if (!b) return;
    const txt = b.innerHTML;
    b.innerHTML = t("copied");
    setTimeout(() => (b.innerHTML = txt), 1200);
  });

  on("btnChooseCSV", "click", () =>
    document.getElementById("csvFile")?.click()
  );
  on("btnImportJSON", "click", () =>
    document.getElementById("importJSON")?.click()
  );

  on("btnLangToggle", "click", () => {
    setLang(getLang() === "it" ? "en" : "it");
  });

  const lb = (idIn, idOut) => {
    const input = document.getElementById(idIn);
    const output = document.getElementById(idOut);
    if (!input || !output) return;
    output.textContent = input.value;
    input.addEventListener("input", (e) => {
      output.textContent = e.target.value;
    });
  };
  lb("lr", "lrVal");
  lb("epochs", "epochsVal");
  lb("batch", "batchVal");

  document.querySelectorAll(".palette-item").forEach((el) => {
    el.addEventListener("dragstart", (ev) => {
      ev.dataTransfer.setData("text/plain", el.dataset.type);
    });
  });

  const importJSON = document.getElementById("importJSON");
  if (importJSON && !importJSON._wiredJson) {
    importJSON._wiredJson = true;
    importJSON.addEventListener("change", (e) => {
      const input = e.target;
      const file = input.files?.[0];
      handleJSONImportFile(file, input);
    });
  }
}

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
    "btnLangToggle",
    "csvFile",
    "presetDataset",
    "csvInfoBtn",
  ];

  console.group("%c[NeuroBuilder] Check UI", "color:#0ea5e9;font-weight:700");
  ids.forEach((id) => {
    const el = document.getElementById(id);
    console[el ? "log" : "warn"](
      `${el ? "✓" : "✗"} ${id} ${el ? "found" : "missing"}`
    );
  });
  console.groupEnd();
}

// ========= Init =========
document.addEventListener("DOMContentLoaded", () => {
  const savedLang = localStorage.getItem("neurobuilder-lang");
  if (savedLang === "it" || savedLang === "en") {
    document.body.dataset.lang = savedLang;
    document.documentElement.lang = savedLang;
  } else if (!document.body.dataset.lang) {
    document.body.dataset.lang = "it";
    document.documentElement.lang = "it";
  }

  sanityCheckButtons();
  attachArchDnD();
  attachCsvDnD();
  observeCsvInputs();

  arch = [];
  addLayer("input");
  addLayer("hidden");
  addLayer("output");
  buildNetwork();

  bindUIControls();
  wireCsvInputs();
  attachPopoverGlobalClosers();
  applyI18n();
  initCsvInfoSafe();
});
