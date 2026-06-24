// ========= Visualization =========
let lastIOLayout = null;
let ioOverlayResizeObserver = null;

function syncPredictionOverlay() {
  const svg = $("#nnVis");
  const canvas = svg?.closest(".network-canvas");
  const inputsBox = $("#testInputs");
  const outputsBox = $("#predictOutputs");
  if (!svg || !canvas || !inputsBox || !outputsBox) return;

  if (!ioOverlayResizeObserver && typeof ResizeObserver !== "undefined") {
    ioOverlayResizeObserver = new ResizeObserver(() => syncPredictionOverlay());
    ioOverlayResizeObserver.observe(canvas);
  }

  const isTextMode = inputConfig.mode === "text";
  inputsBox.classList.toggle("io-overlay-static", isTextMode);
  outputsBox.classList.toggle("io-overlay-static", isTextMode);
  if (isTextMode || !lastIOLayout) return;

  const svgRect = svg.getBoundingClientRect();
  const canvasRect = canvas.getBoundingClientRect();
  if (!svgRect.width || !svgRect.height) return;

  // L'svg usa il preserveAspectRatio di default "xMidYMid meet": la scala è
  // uniforme (il minimo tra i due rapporti) e il contenuto viene centrato
  // nello spazio in eccesso sull'asse non limitante.
  const scale = Math.min(
    svgRect.width / lastIOLayout.W,
    svgRect.height / lastIOLayout.H,
  );
  const offsetX =
    svgRect.left - canvasRect.left + (svgRect.width - lastIOLayout.W * scale) / 2;
  const offsetY =
    svgRect.top - canvasRect.top + (svgRect.height - lastIOLayout.H * scale) / 2;

  Array.from(inputsBox.children).forEach((el, i) => {
    const node = lastIOLayout.inputs[i];
    if (!node) return;
    el.style.left = `${offsetX + node.x * scale}px`;
    el.style.top = `${offsetY + node.y * scale}px`;
  });

  Array.from(outputsBox.children).forEach((el, i) => {
    const node = lastIOLayout.outputs[i];
    if (!node) return;
    el.style.left = `${offsetX + node.x * scale}px`;
    el.style.top = `${offsetY + node.y * scale}px`;
  });
}

function formatDebugValues(values, maxItems = 6) {
  if (!Array.isArray(values)) return "-";
  const shown = values.slice(0, maxItems).map((value) =>
    Number.isFinite(value) ? Number(value.toFixed(3)).toString() : "-",
  );
  return `[${shown.join(", ")}${values.length > maxItems ? ", …" : ""}]`;
}

function debugExampleLabel(maxLength = 68) {
  if (!stepTrainingState.active) return "";
  if (stepTrainingState.rawInput) {
    const text = stepTrainingState.rawInput;
    return text.length > maxLength
      ? `${text.slice(0, Math.max(1, maxLength - 3))}…`
      : text;
  }
  return formatDebugValues(stepTrainingState.input);
}

function renderStepDebugPanel(width, compact = false, offsetY = 0) {
  if (!stepTrainingState.active) return "";

  const target = formatDebugValues(stepTrainingState.target, 3);
  const output = formatDebugValues(stepTrainingState.output, 3);
  const loss = formatStepNumber(stepTrainingState.loss);
  const delta = stepTrainingState.trace
    ? formatStepNumber(stepTrainingState.trace.meanAbsDelta, 6)
    : "-";

  if (compact) {
    return `
      <g class="step-debug-panel">
        <rect x="8" y="${offsetY}" width="${width - 16}" height="122" rx="6"
          fill="#07111f" stroke="rgba(56,189,248,.55)" stroke-width="1.2" />
        <text x="18" y="${offsetY + 21}" class="step-debug-phase">${escapeHTML(stepPhaseLabel())}</text>
        <text x="18" y="${offsetY + 43}" class="step-debug-example"><tspan class="step-debug-key">${escapeHTML(t("stepExample"))}:</tspan> ${escapeHTML(debugExampleLabel(42))}</text>
        <text x="18" y="${offsetY + 65}" class="step-debug-metric"><tspan class="step-debug-key">${escapeHTML(t("stepDesired"))}:</tspan> ${escapeHTML(target)}</text>
        <text x="18" y="${offsetY + 87}" class="step-debug-metric"><tspan class="step-debug-key">${escapeHTML(t("stepCurrent"))}:</tspan> ${escapeHTML(output)}</text>
        <text x="18" y="${offsetY + 109}" class="step-debug-metric"><tspan class="step-debug-key">Loss:</tspan> ${escapeHTML(loss)} <tspan dx="24" class="step-debug-key">|Δw|:</tspan> ${escapeHTML(delta)}</text>
      </g>`;
  }

  return `
    <g class="step-debug-panel">
      <rect x="18" y="${offsetY}" width="764" height="76" rx="6"
        fill="#07111f" stroke="rgba(56,189,248,.55)" stroke-width="1.2" />
      <text x="34" y="${offsetY + 22}" class="step-debug-phase">${escapeHTML(stepPhaseLabel())}</text>
      <text x="34" y="${offsetY + 44}" class="step-debug-example">
        <tspan class="step-debug-key">${escapeHTML(t("stepExample"))}:</tspan>
        ${escapeHTML(debugExampleLabel())}
      </text>
      <text x="34" y="${offsetY + 66}" class="step-debug-metric"><tspan class="step-debug-key">${escapeHTML(t("stepDesired"))}:</tspan> ${escapeHTML(target)}</text>
      <text x="270" y="${offsetY + 66}" class="step-debug-metric"><tspan class="step-debug-key">${escapeHTML(t("stepCurrent"))}:</tspan> ${escapeHTML(output)}</text>
      <text x="548" y="${offsetY + 66}" class="step-debug-metric"><tspan class="step-debug-key">Loss:</tspan> ${escapeHTML(loss)}</text>
      <text x="674" y="${offsetY + 66}" class="step-debug-metric"><tspan class="step-debug-key">|Δw|:</tspan> ${escapeHTML(delta)}</text>
    </g>`;
}

function renderNNVis() {
  const svg = $("#nnVis");
  if (!svg) return;

  const compact = window.innerWidth < 600;
  const W = compact ? 360 : 800;
  const debugActive = stepTrainingState.active;
  const H = debugActive ? (compact ? 500 : 440) : 360;
  const sizes = [inputSize, ...net.layers.map((L) => L.out)];
  const layerCount = sizes.length;

  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  svg.setAttribute("height", H);

  if (layerCount < 1) {
    svg.innerHTML = "";
    return;
  }

  const xPad = compact ? 38 : 80;
  const yPad = 30;
  const colW = (W - 2 * xPad) / (layerCount - 1 || 1);
  const largestLayer = Math.max(...sizes);
  const debugPanelSpace = debugActive ? (compact ? 140 : 100) : 30;
  const networkHeight = H - yPad - debugPanelSpace;
  const availableNodeGap = networkHeight / (largestLayer + 1);
  const nodeR = Math.max(1.5, Math.min(13, availableNodeGap * 0.36));
  const nodeLabelSize = nodeR < 3 ? 0 : Math.max(5, Math.min(11, nodeR));
  const nodeStrokeWidth = Math.max(0.6, Math.min(1.6, nodeR * 0.4));
  const BIAS_ROW_Y = 40;
  const debugPanelY = debugActive ? H - (compact ? 130 : 88) : 0;

  const pos = [];
  for (let li = 0; li < layerCount; li++) {
    const n = sizes[li];
    const totalH = networkHeight;
    const gap = totalH / (n + 1);
    const x = xPad + li * colW;
    for (let ni = 0; ni < n; ni++) {
      const y = yPad + (ni + 1) * gap;
      pos.push({ li, ni, x, y });
    }
  }

  lastIOLayout = {
    W,
    H,
    inputs: pos.filter((p) => p.li === 0),
    outputs: pos.filter((p) => p.li === layerCount - 1),
  };

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
    <filter id="edgeGlow" filterUnits="userSpaceOnUse" x="-50" y="-50" width="${W + 100}" height="${H + 100}">
      <feGaussianBlur stdDeviation="1.2" result="blur"/>
      <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
    </filter>
    <marker id="forwardArrow" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto">
      <path d="M0,0 L0,6 L8,3 z" fill="#38bdf8" />
    </marker>
    <marker id="backwardArrow" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto">
      <path d="M0,0 L0,6 L8,3 z" fill="#fbbf24" />
    </marker>
  </defs>`;

  let edges = "";
  let directionEdges = "";
  let biasNodes = "";

  net.layers.forEach((Lyr, li) => {
    // ==========================
    // ARCHI PESI NORMALI
    // ==========================

    for (let i = 0; i < Lyr.W.length; i++) {
      for (let j = 0; j < Lyr.W[0].length; j++) {
        const from = pos[nodeIndex(li, i)];
        const to = pos[nodeIndex(li + 1, j)];

        if (!from || !to) continue;

        const w = Lyr.W[i][j];
        const trace =
          stepTrainingState.phase === "backward" &&
          stepTrainingState.backwardLayer === li
            ? stepTrainingState.trace
            : null;
        const delta = trace?.weightDeltas?.[i]?.[j];
        const previousWeight = trace?.weightsBefore?.[i]?.[j];
        const forwardActive =
          stepTrainingState.phase === "forward" &&
          stepTrainingState.forwardLayer === li;
        const backwardActive = !!trace;
        const directionalStep =
          stepTrainingState.phase === "forward" ||
          stepTrainingState.phase === "backward";
        const sw = 1 + 5 * (Math.abs(w) / maxAbs);
        const stroke = w >= 0 ? "#22c55e" : "#ef4444";
        const cx = (from.x + to.x) / 2;
        const cy = from.y + (to.y - from.y) * 0.1;

        edges += `
<g class="edge-group"
   data-weight="${w.toFixed(4)}"
   data-previous-weight="${Number.isFinite(previousWeight) ? previousWeight.toFixed(4) : ""}"
   data-weight-delta="${Number.isFinite(delta) ? delta.toFixed(6) : ""}">

  <path
    class="edge-hitbox"
    d="M ${from.x},${from.y} Q ${cx},${cy} ${to.x},${to.y}"
    stroke="transparent"
    stroke-width="${Math.max(sw + 12, 16)}"
    fill="none"
  />

  <path
    d="M ${from.x},${from.y} Q ${cx},${cy} ${to.x},${to.y}"
    stroke="${stroke}"
    stroke-width="${sw}"
    stroke-linecap="round"
    fill="none"
    opacity="${directionalStep && !forwardActive && !backwardActive ? "0.24" : "0.95"}"
    filter="url(#edgeGlow)"
  />
</g>`;

        if (forwardActive) {
          directionEdges += `<path class="step-flow step-flow-forward"
            d="M ${from.x},${from.y} Q ${cx},${cy} ${to.x},${to.y}"
            marker-end="url(#forwardArrow)" />`;
        } else if (backwardActive) {
          directionEdges += `<path class="step-flow step-flow-backward"
            d="M ${to.x},${to.y} Q ${cx},${cy} ${from.x},${from.y}"
            marker-end="url(#backwardArrow)" />`;
        }
      }
    }

    // ==========================
    // NODO BIAS DEL LAYER
    // ==========================

    if (Lyr.useBias && Lyr.b && Lyr.b[0]) {
      const targetLayerIndex = li + 1;

      const layerNodes = pos.filter((p) => p.li === targetLayerIndex);
      if (!layerNodes.length) return;

      const minY = Math.min(...layerNodes.map((p) => p.y));

      const biasX = xPad + targetLayerIndex * colW - colW * 0.35;
      const biasY = BIAS_ROW_Y;

      const biasLabel = `b${targetLayerIndex}`;

      biasNodes += `
        <g class="nn-bias-node">
          <circle
            cx="${biasX}"
            cy="${biasY}"
            r="${nodeR}"
            style="fill:#312e81; stroke:rgba(255,255,255,0.95); stroke-width:1.6"
          />
          <text
            x="${biasX}"
            y="${biasY + 4}"
            text-anchor="middle"
            style="fill:#e5e7eb;font-weight:700;font-size:11px"
          >
            ${biasLabel}
          </text>
        </g>`;

      for (let j = 0; j < Lyr.out; j++) {
        const to = pos[nodeIndex(targetLayerIndex, j)];
        if (!to) continue;

        const b = Lyr.b[0][j] ?? 0;
        const biasTrace =
          stepTrainingState.phase === "backward" &&
          stepTrainingState.backwardLayer === li
            ? stepTrainingState.trace
            : null;
        const biasDelta = biasTrace?.biasDeltas?.[j];
        const previousBias = biasTrace?.biasBefore?.[j];
        const forwardActive =
          stepTrainingState.phase === "forward" &&
          stepTrainingState.forwardLayer === li;
        const backwardActive = !!biasTrace;
        const directionalStep =
          stepTrainingState.phase === "forward" ||
          stepTrainingState.phase === "backward";
        const sw = 1 + 5 * (Math.abs(b) / maxAbs);
        const stroke = b >= 0 ? "#22c55e" : "#ef4444";

        const cx = (biasX + to.x) / 2;
        const cy = biasY + (to.y - biasY) * 0.15;

        edges += `
          <g class="edge-group"
            data-weight="${b.toFixed(4)}"
            data-previous-weight="${Number.isFinite(previousBias) ? previousBias.toFixed(4) : ""}"
            data-weight-delta="${Number.isFinite(biasDelta) ? biasDelta.toFixed(6) : ""}">

            <path
              class="edge-hitbox"
              d="M ${biasX},${biasY} Q ${cx},${cy} ${to.x},${to.y}"
              stroke="transparent"
              stroke-width="${Math.max(sw + 12, 16)}"
              fill="none"
            />

            <path
              d="M ${biasX},${biasY} Q ${cx},${cy} ${to.x},${to.y}"
              stroke="${stroke}"
              stroke-width="${sw}"
              stroke-linecap="round"
              fill="none"
              opacity="${directionalStep && !forwardActive && !backwardActive ? "0.2" : "0.7"}"
              stroke-dasharray="4 4"
              filter="url(#edgeGlow)"
            />
          </g>`;

        if (forwardActive) {
          directionEdges += `<path class="step-flow step-flow-forward"
            d="M ${biasX},${biasY} Q ${cx},${cy} ${to.x},${to.y}"
            marker-end="url(#forwardArrow)" />`;
        } else if (backwardActive) {
          directionEdges += `<path class="step-flow step-flow-backward"
            d="M ${to.x},${to.y} Q ${cx},${cy} ${biasX},${biasY}"
            marker-end="url(#backwardArrow)" />`;
        }
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
      const vZ = lastNodeColors?.z?.[li]?.[ni];

      let fill = "#0b1220";
      if (vNorm != null) {
        const tt = clamp01(vNorm + 0.15);

        if (isInput) {
          fill = `hsl(${Math.round(210 - tt * 160)}, 85%, 55%)`;
        } else if (isOutput) {
          fill = outputColor(tt);
        } else {
          fill = hiddenColor(tt);
        }
      }

      const textInputLabels =
        inputConfig.mode === "text" ? [...textAlphabetCharacters(), "?"] : [];
      const label = isInput
        ? inputConfig.mode === "text"
          ? textInputLabels[ni] || "?"
          : "x" + (ni + 1)
        : isOutput
          ? "y" + (ni + 1)
          : "h" + (ni + 1);

      const desiredBadge =
        isOutput && stepTrainingState.active && stepTrainingState.target?.[ni] != null
          ? `<text x="${p.x}" y="${p.y + nodeR + 15}" text-anchor="middle"
              class="step-output-target">${escapeHTML(t("stepTargetShort"))}: ${formatStepNumber(
                stepTrainingState.target[ni],
              )}</text>`
          : "";

      nodes += `<g class="nn-node"
  data-activation="${vRaw ?? ""}"
  data-z="${!isInput && vZ != null ? vZ : ""}">
        <circle cx="${p.x}" cy="${p.y}" r="${nodeR}"
          style="fill:${fill} !important; stroke:rgba(255,255,255,0.95); stroke-width:${nodeStrokeWidth}"/>
        <text x="${p.x}" y="${p.y + Math.min(3, nodeR * 0.3)}" text-anchor="middle"
          style="fill:#e5e7eb;font-weight:600;font-size:${nodeLabelSize}px">${escapeHTML(label)}</text>
        ${desiredBadge}
        <circle class="node-hitbox" cx="${p.x}" cy="${p.y}" r="${nodeR + 10}"
          style="fill:transparent !important; stroke:none !important"/>
      </g>`;
    }
  }

  svg.innerHTML =
    defs +
    `<g>${edges}</g><g>${directionEdges}</g><g>${nodes}</g><g>${biasNodes}</g>` +
    renderStepDebugPanel(W, compact, debugPanelY);
  const tooltip = document.getElementById("edgeTooltip");

  svg.querySelectorAll(".edge-group").forEach((g) => {
    g.addEventListener("mousemove", (e) => {
      const weight = g.dataset.weight;
      const previousWeight = g.dataset.previousWeight;
      const weightDelta = g.dataset.weightDelta;
      const change =
        previousWeight && weightDelta
          ? `<div class="edge-tooltip-change">${previousWeight} → ${weight}<br>Δ ${weightDelta}</div>`
          : "";

      tooltip.innerHTML = `
        <div class="tooltip-label">${t("weight")}</div>
        <div class="tooltip-value">${weight}</div>
        ${change}
      `;

      tooltip.style.left = e.clientX + 16 + "px";
      tooltip.style.top = e.clientY + 16 + "px";

      tooltip.style.opacity = "1";
      tooltip.style.transform = "translateY(0)";
    });

    g.addEventListener("mouseleave", () => {
      tooltip.style.opacity = "0";
      tooltip.style.transform = "translateY(4px)";
    });
  });
  svg.querySelectorAll(".nn-node").forEach((node) => {
    const value = node.dataset.activation;
    const zValue = node.dataset.z;

    if (value === undefined) return;

    node.addEventListener("mousemove", (e) => {
      const zRow =
        zValue !== undefined && zValue !== ""
          ? `
            <div class="tooltip-row">
              <div class="tooltip-label">${t("weightedSum")}</div>
              <div class="tooltip-value">${Number(zValue).toFixed(4)}</div>
            </div>
          `
          : "";
      tooltip.innerHTML = `
        <div class="tooltip-row">
          <div class="tooltip-label">${t("activation")}</div>
          <div class="tooltip-value">${Number(value).toFixed(4)}</div>
        </div>
        ${zRow}
      `;

      tooltip.style.left = e.clientX + 16 + "px";
      tooltip.style.top = e.clientY + 16 + "px";

      tooltip.style.opacity = "1";
      tooltip.style.transform = "translateY(0)";
    });

    node.addEventListener("mouseleave", () => {
      tooltip.style.opacity = "0";
      tooltip.style.transform = "translateY(4px)";
    });
  });

  syncPredictionOverlay();
}

// ========= Build Network =========
function buildNetwork() {
  console.log("BUILD ARCH", arch);

  if (typeof resetStepTraining === "function") {
    resetStepTraining({ render: false });
  }

  net = new Network();

  const rand = rng(Date.now() + rebuildSeedCounter++);

  const inputLayer = arch.find((l) => l.type === "input");

  inputSize = inputLayer?.neurons || 1;

  let previousNeurons = inputSize;

  for (let i = 1; i < arch.length; i++) {
    const layer = arch[i];

    if (layer.type === "input") continue;

    net.add(
      new DenseLayer(
        previousNeurons,
        layer.neurons,
        layer.activation || (layer.type === "output" ? "sigmoid" : "relu"),
        layer.bias ?? true,
        rand,
      ),
    );

    previousNeurons = layer.neurons;
  }

  const outputLayer = arch.find((l) => l.type === "output");

  outputSize = outputLayer?.neurons || 1;

  lastNodeColors = null;

  renderTestInputs();
  renderNNVis();
  updateJSON();
}
