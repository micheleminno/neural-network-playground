// ========= Test Inputs UI =========
function renderTestInputs() {
  const container = $("#testInputs");
  if (!container) return;

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
function renderArchitecture() {
  const archEl = $("#architecture");
  if (!archEl) return;

  archEl.innerHTML = "";

  if (!arch.length) {
    archEl.innerHTML = `
      <div class="text-center small-muted py-4">
        ${t("emptyArchitecture")}
      </div>
    `;
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
    card.className = "layer-card mb-2";

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
            ${isInput ? t("inputSize") : t("neurons")}:
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

    const neuronInput = card.querySelector("[data-neurons]");

    neuronInput.addEventListener("input", (e) => {
      const i = Number(e.target.dataset.neurons);

      arch[i].neurons = Number(e.target.value);

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
