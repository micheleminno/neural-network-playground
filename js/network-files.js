// ========= JSON Export/Import & sync =========
function serializeArchitectureForExport() {
  return {
    inputConfig: serializeInputConfig(),
    layers: arch.map((l) => ({
      type: l.type,
      neurons: l.neurons,
      activation: l.activation,
      bias: l.bias,
    })),
  };
}

function serializeWeightsForExport() {
  return net.layers.map((l) => ({
      in: l.in,
      out: l.out,
      activation: l.activation,
      useBias: l.useBias,
      W: l.W,
      b: l.b,
    }));
}

function getExportPayload(mode = "full") {
  if (mode === "architecture") {
    return {
      data: { architecture: serializeArchitectureForExport() },
      filename: "neurobuilder-architecture.json",
    };
  }

  if (mode === "weights") {
    return {
      data: {
        inputConfig: serializeInputConfig(),
        layers: serializeWeightsForExport(),
      },
      filename: "neurobuilder-weights.json",
    };
  }

  return {
    data: {
      architecture: serializeArchitectureForExport(),
      weights: serializeWeightsForExport(),
      dataset: {
        X: dataset.X,
        y: dataset.y,
        rawText: dataset.rawText,
      },
    },
    filename: "neurobuilder-full-network.json",
  };
}

function getSelectedExportMode() {
  return document.getElementById("exportMode")?.value || "full";
}

function getJSONPreviewData() {
  return getExportPayload(getSelectedExportMode()).data;
}

function summarizeJSONValue(value) {
  if (Array.isArray(value)) {
    const label = t(value.length === 1 ? "jsonItem" : "jsonItems");
    return `[... ${value.length} ${label}]`;
  }
  if (value && typeof value === "object") {
    const count = Object.keys(value).length;
    const label = t(count === 1 ? "jsonProperty" : "jsonProperties");
    return `{... ${count} ${label}}`;
  }
  return JSON.stringify(value);
}

function isJSONBranch(value) {
  return Boolean(value && typeof value === "object");
}

function jsonPathKey(path) {
  return JSON.stringify(path);
}

function appendJSONTreeRow(container, value, key, path, depth, isLast) {
  const row = document.createElement("div");
  row.className = "json-tree-row";
  row.style.setProperty("--json-depth", depth);
  row.setAttribute("role", "treeitem");

  const keyLabel = document.createElement("span");
  keyLabel.className = "json-tree-key";
  keyLabel.textContent =
    typeof key === "number" ? `[${key}]` : `${JSON.stringify(key)}`;

  if (!isJSONBranch(value)) {
    row.appendChild(keyLabel);
    row.append(":");
    const primitive = document.createElement("span");
    primitive.className = "json-tree-primitive";
    primitive.textContent = `${JSON.stringify(value)}${isLast ? "" : ","}`;
    row.appendChild(primitive);
    container.appendChild(row);
    return;
  }

  const pathKey = jsonPathKey(path);
  const expanded = jsonExpandedPaths.has(pathKey);
  const toggle = document.createElement("button");
  toggle.type = "button";
  toggle.className = "json-tree-toggle";
  toggle.textContent = expanded ? "−" : "+";
  toggle.setAttribute("aria-expanded", String(expanded));
  toggle.setAttribute("aria-label", t(expanded ? "jsonCollapseNode" : "jsonExpandNode"));
  toggle.addEventListener("click", () => {
    if (expanded) jsonExpandedPaths.delete(pathKey);
    else jsonExpandedPaths.add(pathKey);
    renderJSONTree(getJSONPreviewData());
  });

  row.append(toggle, keyLabel);
  row.append(":");

  if (!expanded) {
    const summary = document.createElement("span");
    summary.className = "json-tree-summary";
    summary.textContent = `${summarizeJSONValue(value)}${isLast ? "" : ","}`;
    row.appendChild(summary);
    container.appendChild(row);
    return;
  }

  row.append(` ${Array.isArray(value) ? "[" : "{"}`);
  container.appendChild(row);

  const entries = Array.isArray(value)
    ? value.map((child, index) => [index, child])
    : Object.entries(value);
  entries.forEach(([childKey, child], index) => {
    appendJSONTreeRow(
      container,
      child,
      childKey,
      [...path, childKey],
      depth + 1,
      index === entries.length - 1,
    );
  });

  const closing = document.createElement("div");
  closing.className = "json-tree-row";
  closing.style.setProperty("--json-depth", depth);
  closing.textContent = `${Array.isArray(value) ? "]" : "}"}${isLast ? "" : ","}`;
  container.appendChild(closing);
}

function renderJSONTree(value) {
  const tree = document.getElementById("jsonTree");
  if (!tree) return;
  tree.replaceChildren();

  const opening = document.createElement("div");
  opening.className = "json-tree-row";
  opening.textContent = "{";
  tree.appendChild(opening);

  const entries = Object.entries(value);
  entries.forEach(([key, child], index) => {
    appendJSONTreeRow(
      tree,
      child,
      key,
      [key],
      1,
      index === entries.length - 1,
    );
  });

  const closing = document.createElement("div");
  closing.className = "json-tree-row";
  closing.textContent = "}";
  tree.appendChild(closing);
}

function updateJSON() {
  const j = getJSONPreviewData();

  const jsonArea = $("#jsonArea");
  const jsonTree = $("#jsonTree");
  const compactControl = $("#jsonCompact");
  if (!jsonArea || !jsonTree) return;

  jsonArea.classList.toggle("d-none", jsonCollapsed);
  jsonTree.classList.toggle("d-none", !jsonCollapsed);
  if (compactControl) compactControl.disabled = jsonCollapsed;

  if (jsonCollapsed) {
    renderJSONTree(j);
  } else {
    jsonArea.value = jsonCompact ? JSON.stringify(j) : JSON.stringify(j, null, 2);
  }
}

function handleJSONImportFile(file, inputEl) {
  if (!file) return;

  const fr = new FileReader();

  fr.onload = () => {
    try {
      const o = JSON.parse(fr.result);

      // ========= PESI =========

      if (Array.isArray(o.layers)) {
        if (!o.layers.length) throw new Error(t("layersEmpty"));

        arch = [
          {
            id: crypto.randomUUID(),
            type: "input",
            neurons: o.layers[0].in,
          },

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
            neurons: o.layers.at(-1).out,
            activation: o.layers.at(-1).activation,
            bias: o.layers.at(-1).useBias,
          },
        ];

        inputSize = arch[0].neurons;
        outputSize = arch[arch.length - 1].neurons;
        applyInputConfig(o.inputConfig, inputSize);
        if (inputConfig.mode === "text" && textFeatureCount() !== inputSize) {
          throw new Error(t("textConfigMismatch"));
        }
        dataset = { X: [], y: [], rawText: [] };
        renderDatasetPreview();

        buildNetwork();

        for (let i = 0; i < net.layers.length; i++) {
          net.layers[i].W = o.layers[i].W;
          net.layers[i].b = o.layers[i].b;
        }

        renderArchitecture();
        renderTestInputs();
        renderNNVis();
        syncInputModeControls();
        updateJSON();
        predictOnce(true);

        alert(t("importWeightsOk"));
      }

      // ========= ARCHITETTURA =========
      else if (o.architecture || o.weights) {
        if (!o.architecture?.layers) throw new Error(t("architectureMissing"));

        arch = o.architecture.layers.map((l) => ({
          id: l.id || crypto.randomUUID(),
          type: l.type,
          neurons: l.neurons,
          activation: l.activation,
          bias: l.bias,
        }));

        inputSize = arch.find((l) => l.type === "input")?.neurons;

        outputSize = arch.find((l) => l.type === "output")?.neurons;
        applyInputConfig(o.inputConfig || o.architecture.inputConfig, inputSize);
        if (inputConfig.mode === "text" && textFeatureCount() !== inputSize) {
          throw new Error(t("textConfigMismatch"));
        }

        buildNetwork();

        if (
          Array.isArray(o.weights) &&
          o.weights.length === net.layers.length
        ) {
          for (let i = 0; i < net.layers.length; i++) {
            if (o.weights[i].W) {
              net.layers[i].W = o.weights[i].W;
            }

            if (o.weights[i].b !== undefined) {
              net.layers[i].b = o.weights[i].b;
            }
          }
        }

        if (o.dataset) {
          dataset.X = Array.isArray(o.dataset.X) ? o.dataset.X : [];
          dataset.y = Array.isArray(o.dataset.y) ? o.dataset.y : [];
          dataset.rawText = Array.isArray(o.dataset.rawText)
            ? o.dataset.rawText
            : [];
          if (inputConfig.mode === "text" && dataset.rawText.length) {
            updateTextDatasetEncoding();
          }
        } else {
          dataset = { X: [], y: [], rawText: [] };
        }

        renderDatasetPreview();

        renderArchitecture();
        renderTestInputs();
        renderNNVis();
        syncInputModeControls();
        updateJSON();
        predictOnce(true);

        alert(
          t("importArchOk") +
            (o.weights ? t("importWeightsSuffix") : "") +
            t("importClose"),
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

function exportJSONFile(mode = "full") {
  const { data, filename } = getExportPayload(mode);

  const blob = new Blob([JSON.stringify(data, null, 2)], {
    type: "application/json",
  });

  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = filename;
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
    getLang() === "it" ? "Informazioni formato CSV" : "CSV format info",
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
    true,
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
    true,
  );
}
