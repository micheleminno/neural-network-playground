function freshStepTrainingState(sampleCursor = -1) {
  return {
    active: false,
    phase: "idle",
    sampleCursor,
    sampleIndex: -1,
    input: null,
    rawInput: "",
    target: null,
    output: null,
    loss: null,
    activations: [],
    preActivations: [],
    forwardLayer: -1,
    backwardLayer: -1,
    backwardGradient: null,
    trace: null,
    completedUpdates: 0,
  };
}

function resetStepTraining({ render = true, preserveCursor = false } = {}) {
  const cursor = preserveCursor ? stepTrainingState.sampleCursor : -1;
  stepTrainingState = freshStepTrainingState(cursor);
  lastNodeColors = null;
  updateStepTrainingControls();

  if (render) {
    renderNNVis();
  }
}

function setTrainingMode(mode) {
  const stepMode = mode === "step";
  const standard = document.getElementById("trainingModeStandard");
  const step = document.getElementById("trainingModeStep");
  const standardActions = document.getElementById("standardTrainingActions");
  const stepActions = document.getElementById("stepTrainingActions");
  const epochs = document.getElementById("epochs");
  const batch = document.getElementById("batch");

  if (standard) standard.checked = !stepMode;
  if (step) step.checked = stepMode;
  if (stepMode) stopFlag = true;
  standardActions?.classList.toggle("d-none", stepMode);
  stepActions?.classList.toggle("d-none", !stepMode);
  if (epochs) epochs.disabled = stepMode;
  if (batch) batch.disabled = stepMode;
  resetStepTraining();
}

function formatStepNumber(value, digits = 4) {
  if (!Number.isFinite(value)) return "-";
  return Number(value.toFixed(digits)).toString();
}

function stepPhaseLabel() {
  const state = stepTrainingState;

  if (!state.active || state.phase === "idle") return t("stepReady");
  if (state.phase === "loaded") return t("stepExampleLoaded");
  if (state.phase === "forward") {
    return t("stepForwardLayer").replace("{layer}", state.forwardLayer + 1);
  }
  if (state.phase === "loss") return t("stepLossCalculated");
  if (state.phase === "backward") {
    return t("stepBackwardLayer").replace("{layer}", state.backwardLayer + 1);
  }
  if (state.phase === "complete") return t("stepUpdateComplete");
  return t("stepReady");
}

function updateStepTrainingControls() {
  const status = document.getElementById("stepTrainingStatus");
  const next = document.getElementById("btnStepNext");
  const sample = stepTrainingState.sampleIndex + 1;

  if (status) {
    status.textContent = stepTrainingState.active
      ? `${stepPhaseLabel()} · ${t("stepExampleCounter")} ${sample}/${dataset.X.length}`
      : t("stepReady");
  }

  if (next) {
    next.innerHTML =
      stepTrainingState.phase === "idle" || stepTrainingState.phase === "complete"
        ? `<i class="bi bi-box-arrow-in-down"></i> ${t("stepLoadExample")}`
        : `<i class="bi bi-skip-forward"></i> ${t("stepNext")}`;
  }
}

function setStepNodeColors() {
  const activations = stepTrainingState.activations;
  if (!activations.length) {
    lastNodeColors = null;
    return;
  }

  lastNodeColors = { byLayer: [], raw: [], z: [] };
  const preActivations = stepTrainingState.preActivations || [];
  activations.forEach((values, layerIndex) => {
    if (!values) return;
    lastNodeColors.raw[layerIndex] = values.slice();
    if (preActivations[layerIndex]) {
      lastNodeColors.z[layerIndex] = preActivations[layerIndex].slice();
    }

    if (values.length === 1) {
      lastNodeColors.byLayer[layerIndex] = [clamp01(values[0])];
      return;
    }

    const min = Math.min(...values);
    const max = Math.max(...values);
    const range = max - min || 1;
    lastNodeColors.byLayer[layerIndex] = values.map((value) =>
      clamp01((value - min) / range),
    );
  });
}

function loadNextStepExample() {
  const nextIndex = (stepTrainingState.sampleCursor + 1) % dataset.X.length;
  const rawInput = dataset.rawText?.[nextIndex] || "";
  stepTrainingState = freshStepTrainingState(nextIndex);
  stepTrainingState.active = true;
  stepTrainingState.phase = "loaded";
  stepTrainingState.sampleIndex = nextIndex;
  stepTrainingState.input = dataset.X[nextIndex].slice();
  stepTrainingState.rawInput = rawInput;
  stepTrainingState.target = dataset.y[nextIndex].slice();
  stepTrainingState.activations = [stepTrainingState.input.slice()];
  stepTrainingState.preActivations = [];
  setStepNodeColors();
}

function runNextForwardLayer() {
  const layerIndex = stepTrainingState.forwardLayer + 1;
  const layer = net.layers[layerIndex];
  const previousActivation = stepTrainingState.activations[layerIndex];
  const activation = layer.forward([previousActivation])[0];

  stepTrainingState.phase = "forward";
  stepTrainingState.forwardLayer = layerIndex;
  stepTrainingState.backwardLayer = -1;
  stepTrainingState.trace = null;
  stepTrainingState.activations[layerIndex + 1] = activation.slice();
  stepTrainingState.preActivations[layerIndex + 1] = layer.Z[0].slice();

  if (layerIndex === net.layers.length - 1) {
    stepTrainingState.output = activation.slice();
  }

  setStepNodeColors();
}

function calculateStepLoss() {
  const [loss, gradient] = bce(
    [stepTrainingState.output],
    [stepTrainingState.target],
  );
  stepTrainingState.phase = "loss";
  stepTrainingState.loss = loss;
  stepTrainingState.backwardGradient = gradient;
  stepTrainingState.trace = null;

  const lossNow = document.getElementById("lossNow");
  if (lossNow) lossNow.textContent = formatStepNumber(loss);
}

function runNextBackwardLayer() {
  const layerIndex =
    stepTrainingState.backwardLayer < 0
      ? net.layers.length - 1
      : stepTrainingState.backwardLayer - 1;
  const layer = net.layers[layerIndex];
  const lr = Number(document.getElementById("lr")?.value ?? 0.1);
  const trace = layer.backwardWithTrace(
    stepTrainingState.backwardGradient,
    lr,
    layerIndex === net.layers.length - 1,
  );

  stepTrainingState.phase = "backward";
  stepTrainingState.backwardLayer = layerIndex;
  stepTrainingState.backwardGradient = trace.dX;
  stepTrainingState.trace = trace;
  stepTrainingState.completedUpdates += 1;
  updateJSON();
}

function completeStepExample() {
  const output = net.forward([stepTrainingState.input])[0];
  const [loss] = bce([output], [stepTrainingState.target]);
  stepTrainingState.phase = "complete";
  stepTrainingState.output = output.slice();
  stepTrainingState.loss = loss;
  stepTrainingState.forwardLayer = net.layers.length - 1;
  stepTrainingState.backwardLayer = -1;
  stepTrainingState.trace = null;
  stepTrainingState.activations = [
    stepTrainingState.input.slice(),
    ...net.layers.map((layer) => layer.A[0].slice()),
  ];
  stepTrainingState.preActivations = [
    undefined,
    ...net.layers.map((layer) => layer.Z[0].slice()),
  ];
  setStepNodeColors();

  const lossNow = document.getElementById("lossNow");
  if (lossNow) lossNow.textContent = formatStepNumber(loss);
}

function nextStepTraining() {
  if (!dataset.X.length || !dataset.y.length) {
    alert(t("trainNoDataset"));
    return;
  }
  if (!net.layers.length) {
    alert(t("stepNoNetwork"));
    return;
  }

  const state = stepTrainingState;
  if (state.phase === "idle" || state.phase === "complete") {
    loadNextStepExample();
  } else if (state.phase === "loaded") {
    runNextForwardLayer();
  } else if (
    state.phase === "forward" &&
    state.forwardLayer < net.layers.length - 1
  ) {
    runNextForwardLayer();
  } else if (state.phase === "forward") {
    calculateStepLoss();
  } else if (state.phase === "loss") {
    runNextBackwardLayer();
  } else if (state.phase === "backward" && state.backwardLayer > 0) {
    runNextBackwardLayer();
  } else if (state.phase === "backward") {
    completeStepExample();
  }

  updateStepTrainingControls();
  renderNNVis();
}
