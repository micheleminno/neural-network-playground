// ========= State =========
let net = new Network();
let arch = [];
let inputSize = 2;
let outputSize = 1;
let dataset = { X: [], y: [], rawText: [] };
let inputConfig = {
  mode: "numeric",
  numericSize: 2,
  text: {
    alphabet: "abcdefghijklmnopqrstuvwxyz ",
    lowercase: true,
    encoding: "frequency",
  },
};
let chart;
let stopFlag = false;
let lastNodeColors = null;
let rebuildSeedCounter = 1;
let currentNetworkName = "";
let currentNetworkId = null;
let jsonCompact = false;
let jsonCollapsed = true;
const jsonExpandedPaths = new Set();
let stepTrainingState = {
  active: false,
  phase: "idle",
  sampleCursor: -1,
  sampleIndex: -1,
  input: null,
  rawInput: "",
  target: null,
  output: null,
  loss: null,
  activations: [],
  forwardLayer: -1,
  backwardLayer: -1,
  backwardGradient: null,
  trace: null,
  completedUpdates: 0,
};

// ========= Colors & Node Coloring =========
const clamp01 = (v) => Math.max(0, Math.min(1, v));
const lerp = (a, b, t) => a + (b - a) * t;
const hiddenColor = (v) =>
  `hsl(210, 80%, ${Math.round(lerp(12, 60, clamp01(v)))}%)`;
const outputColor = (v) => `hsl(${Math.round(120 * clamp01(v))}, 85%, 50%)`;

function computeNodeColorsForInput(xvec) {
  if (!xvec) return;
  net.forward([xvec]);
  lastNodeColors = { byLayer: [], raw: [], z: [] };
  // ===== INPUT LAYER =====

  lastNodeColors.raw[0] = xvec.slice();

  let minIn = Math.min(...xvec);
  let maxIn = Math.max(...xvec);

  const denIn = maxIn - minIn || 1;

  lastNodeColors.byLayer[0] = xvec.map((v) => (v - minIn) / denIn);

  // ===== HIDDEN + OUTPUT =====
  net.layers.forEach((L, k) => {
    const vals = L.A && L.A[0] ? L.A[0].slice() : [];
    lastNodeColors.raw[k + 1] = vals.slice();
    lastNodeColors.z[k + 1] = L.Z && L.Z[0] ? L.Z[0].slice() : [];

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
