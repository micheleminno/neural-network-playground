// ========= Test Inputs UI =========
function renderTestInputs() {
  const container = $("#testInputs");
  if (!container) return;

  if (inputConfig.mode === "text") {
    const previousText = document.getElementById("textPredictInput")?.value || "";
    container.innerHTML = `
      <label class="form-label" for="textPredictInput">${t("textToPredict")}</label>
      <textarea
        id="textPredictInput"
        class="form-control text-predict-input"
        rows="4"
        data-i18n-placeholder="textPredictPlaceholder"
        placeholder="${t("textPredictPlaceholder")}"
      ></textarea>
      <div id="textEncodingPreview" class="small-muted mt-2"></div>
    `;
    const textarea = document.getElementById("textPredictInput");
    textarea.value = previousText;
    textarea.addEventListener("input", () => updateTextEncodingPreview(textarea.value));
    updateTextEncodingPreview(previousText);

    if (typeof renderPredictionOutputs === "function") {
      renderPredictionOutputs();
    }
    return;
  }

  const prev = Array.from(container.querySelectorAll("[data-ti]")).map((inp) =>
    Number(inp.value),
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

  if (typeof renderPredictionOutputs === "function") {
    renderPredictionOutputs();
  }
}

// ========= Architettura =========
const ACTIVATION_PLOT_CONFIG = {
  relu: {
    xMin: -3,
    xMax: 3,
    yMin: -0.4,
    yMax: 3.2,
    formula: "max(0, x)",
  },
  sigmoid: {
    xMin: -6,
    xMax: 6,
    yMin: -0.1,
    yMax: 1.1,
    formula: "1 / (1 + e^-x)",
  },
  tanh: {
    xMin: -3,
    xMax: 3,
    yMin: -1.2,
    yMax: 1.2,
    formula: "tanh(x)",
  },
  linear: {
    xMin: -2,
    xMax: 2,
    yMin: -2.2,
    yMax: 2.2,
    formula: "x",
  },
};

function activationPlotValue(name, x) {
  if (name === "relu") return Math.max(0, x);
  if (name === "sigmoid") return 1 / (1 + Math.exp(-x));
  if (name === "tanh") return Math.tanh(x);
  return x;
}

function createActivationPlot(name) {
  const config = ACTIVATION_PLOT_CONFIG[name] || ACTIVATION_PLOT_CONFIG.linear;
  const width = 180;
  const height = 82;
  const padding = { top: 9, right: 12, bottom: 15, left: 18 };
  const plotWidth = width - padding.left - padding.right;
  const plotHeight = height - padding.top - padding.bottom;
  const mapX = (x) =>
    padding.left + ((x - config.xMin) / (config.xMax - config.xMin)) * plotWidth;
  const mapY = (y) =>
    padding.top + (1 - (y - config.yMin) / (config.yMax - config.yMin)) * plotHeight;
  const svgNs = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(svgNs, "svg");
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  svg.setAttribute("class", "activation-plot");
  svg.setAttribute("role", "img");
  svg.setAttribute("aria-label", `${t("activation")}: ${name}`);

  const addLine = (x1, y1, x2, y2, className) => {
    const line = document.createElementNS(svgNs, "line");
    line.setAttribute("x1", x1);
    line.setAttribute("y1", y1);
    line.setAttribute("x2", x2);
    line.setAttribute("y2", y2);
    line.setAttribute("class", className);
    svg.appendChild(line);
  };

  if (config.yMin <= 0 && config.yMax >= 0) {
    addLine(padding.left, mapY(0), width - padding.right, mapY(0), "activation-axis");
  }
  if (config.xMin <= 0 && config.xMax >= 0) {
    addLine(mapX(0), padding.top, mapX(0), height - padding.bottom, "activation-axis");
  }

  const points = [];
  for (let i = 0; i <= 72; i++) {
    const x = config.xMin + (i / 72) * (config.xMax - config.xMin);
    const y = activationPlotValue(name, x);
    points.push(`${i ? "L" : "M"}${mapX(x).toFixed(2)},${mapY(y).toFixed(2)}`);
  }
  const path = document.createElementNS(svgNs, "path");
  path.setAttribute("d", points.join(" "));
  path.setAttribute("class", "activation-curve");
  svg.appendChild(path);

  const addLabel = (text, x, y, className) => {
    const label = document.createElementNS(svgNs, "text");
    label.textContent = text;
    label.setAttribute("x", x);
    label.setAttribute("y", y);
    label.setAttribute("class", className);
    svg.appendChild(label);
  };
  addLabel("f(x)", 4, 11, "activation-axis-label");
  addLabel("x", width - 10, mapY(0) - 4, "activation-axis-label");
  addLabel(config.formula, width / 2, height - 2, "activation-formula");

  return svg;
}

function renderArchitecture() {
  const archEl = $("#architecture");
  if (!archEl) return;

  const addHiddenControl = document.getElementById("addHiddenControl");
  if (addHiddenControl && archEl.contains(addHiddenControl)) {
    addHiddenControl.remove();
  }

  const inputModePanel = document.getElementById("inputModePanel");
  if (inputModePanel && archEl.contains(inputModePanel)) {
    inputModePanel.remove();
  }
  archEl.innerHTML = "";

  if (!arch.length) {
    archEl.innerHTML = `
      <div class="text-center small-muted py-4">
        ${t("emptyArchitecture")}
      </div>
    `;
    if (addHiddenControl) archEl.appendChild(addHiddenControl);
    return;
  }

  arch.forEach((layerDef, idx) => {
    const isInput = layerDef.type === "input";
    const isOutput = layerDef.type === "output";
    const isHidden = layerDef.type === "hidden";

    const icon =
      layerDef.type === "input"
        ? "bi-box-arrow-in-right text-warning"
        : layerDef.type === "output"
          ? "bi-box-arrow-right text-success"
          : "bi-diagram-3 text-info";

    const name =
      layerDef.type === "input"
        ? t("input")
        : layerDef.type === "output"
          ? t("output")
          : t("hiddenLayer");

    const card = document.createElement("div");
    card.className = `layer-card architecture-layer-card architecture-layer-${layerDef.type}`;

    card.innerHTML = `
      <div class="d-flex align-items-center justify-content-between mb-2">

        <div class="d-flex align-items-center gap-2">
          <i class="bi ${icon}"></i>

          <strong>${name}</strong>

          <span class="badge rounded-pill bg-secondary">
            #${idx + 1}
          </span>
        </div>

        ${
          isHidden
            ? `
              <button
                class="btn btn-sm btn-danger remove-layer"
                type="button"
                title="${t("remove")}"
              >
                <i class="bi bi-x-lg"></i>
              </button>
            `
            : ""
        }
      </div>

      <div class="row g-2">

        <div class="col-6">

          <label class="form-label">
            ${
              isInput
                ? inputConfig.mode === "text"
                  ? t("textFeatures")
                  : t("inputSize")
                : t("neurons")
            }:
            <span id="neuronsVal-${idx}">
              ${layerDef.neurons}
            </span>
          </label>

          <input
            type="number"
            min="1"
            max="64"
            step="1"
            value="${layerDef.neurons}"
            class="form-control"
            data-neurons="${idx}"
            ${isInput && inputConfig.mode === "text" ? "disabled" : ""}
          >

        </div>

        ${
          !isInput
            ? `
              <div class="col-6">

                <label class="form-label">
                  ${t("activation")}
                </label>

                <select
                  class="form-select activation-select"
                  data-activation="${idx}"
                >
                  <option value="relu"
                    ${layerDef.activation === "relu" ? "selected" : ""}>
                    ReLU
                  </option>

                  <option value="sigmoid"
                    ${layerDef.activation === "sigmoid" ? "selected" : ""}>
                    Sigmoid
                  </option>

                  <option value="tanh"
                    ${layerDef.activation === "tanh" ? "selected" : ""}>
                    Tanh
                  </option>

                  <option value="linear"
                    ${layerDef.activation === "linear" ? "selected" : ""}>
                    Linear
                  </option>
                </select>

                <div
                  class="activation-plot-host"
                  data-activation-plot="${idx}"
                ></div>

              </div>

              <div class="col-6 form-check form-switch ms-3">

                <input
                  class="form-check-input bias-switch"
                  type="checkbox"
                  data-bias="${idx}"
                  ${layerDef.bias ? "checked" : ""}
                >

                <label class="form-check-label">
                  ${t("bias")}
                </label>

              </div>
            `
            : ""
        }

      </div>
    `;

    if (isInput && inputModePanel) {
      card.querySelector(".row")?.before(inputModePanel);
    }

    const activationPlotHost = card.querySelector("[data-activation-plot]");
    if (activationPlotHost) {
      activationPlotHost.appendChild(createActivationPlot(layerDef.activation));
    }

    const neuronInput = card.querySelector("[data-neurons]");

    neuronInput.addEventListener("input", (e) => {
      const i = Number(e.target.dataset.neurons);

      arch[i].neurons = Number(e.target.value);

      if (arch[i].type === "input") {
        inputConfig.numericSize = arch[i].neurons;
      }

      const sp = document.getElementById(`neuronsVal-${i}`);
      if (sp) sp.textContent = arch[i].neurons;

      buildNetwork();
      renderArchitecture();
      updateJSON();
    });

    const activationSelect = card.querySelector(".activation-select");

    if (activationSelect) {
      activationSelect.addEventListener("change", (e) => {
        const i = Number(e.target.dataset.activation);

        arch[i].activation = e.target.value;

        buildNetwork();
        renderArchitecture();
        updateJSON();
      });
    }

    const biasSwitch = card.querySelector(".bias-switch");

    if (biasSwitch) {
      biasSwitch.addEventListener("change", (e) => {
        const i = Number(e.target.dataset.bias);

        arch[i].bias = e.target.checked;

        buildNetwork();
        renderArchitecture();
        updateJSON();
      });
    }

    const removeBtn = card.querySelector(".remove-layer");

    if (removeBtn) {
      removeBtn.addEventListener("click", () => {
        arch.splice(idx, 1);

        buildNetwork();
        renderArchitecture();
        updateJSON();
      });
    }

    if (isOutput && addHiddenControl) {
      archEl.appendChild(addHiddenControl);
    }
    archEl.appendChild(card);
  });

  if (addHiddenControl && !archEl.contains(addHiddenControl)) {
    archEl.appendChild(addHiddenControl);
  }
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
  if (type !== "hidden") return;

  arch.splice(arch.length - 1, 0, {
    id: crypto.randomUUID(),
    type: "hidden",
    neurons: 4,
    activation: "relu",
    bias: true,
  });

  buildNetwork();
  renderArchitecture();
  updateJSON();
}
