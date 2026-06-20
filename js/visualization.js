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
  const largestLayer = Math.max(...sizes);
  const availableNodeGap = (H - 2 * yPad) / (largestLayer + 1);
  const nodeR = Math.max(1.5, Math.min(13, availableNodeGap * 0.36));
  const nodeLabelSize = nodeR < 3 ? 0 : Math.max(5, Math.min(11, nodeR));
  const nodeStrokeWidth = Math.max(0.6, Math.min(1.6, nodeR * 0.4));
  const BIAS_ROW_Y = 40;

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
        const sw = 1 + 5 * (Math.abs(w) / maxAbs);
        const stroke = w >= 0 ? "#22c55e" : "#ef4444";
        const cx = (from.x + to.x) / 2;
        const cy = from.y + (to.y - from.y) * 0.1;

        edges += `
<g class="edge-group"
   data-weight="${w.toFixed(4)}">

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
    opacity="0.95"
    filter="url(#edgeGlow)"
  />
</g>`;
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
        const sw = 1 + 5 * (Math.abs(b) / maxAbs);
        const stroke = b >= 0 ? "#22c55e" : "#ef4444";

        const cx = (biasX + to.x) / 2;
        const cy = biasY + (to.y - biasY) * 0.15;

        edges += `
          <g class="edge-group"
            data-weight="${b.toFixed(4)}">

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
              opacity="0.7"
              stroke-dasharray="4 4"
              filter="url(#edgeGlow)"
            />
          </g>`;
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

      const badge =
        !isInput && vRaw != null
          ? `<text x="${p.x}" y="${p.y - (nodeR + 7)}" text-anchor="middle"
              style="fill:#e5e7eb;font-size:10px;font-weight:600">${vRaw.toFixed(
                2,
              )}</text>`
          : "";

      nodes += `<g class="nn-node"
  data-activation="${vRaw ?? ""}">
        <circle cx="${p.x}" cy="${p.y}" r="${nodeR}"
          style="fill:${fill} !important; stroke:rgba(255,255,255,0.95); stroke-width:${nodeStrokeWidth}"/>
        <text x="${p.x}" y="${p.y + Math.min(3, nodeR * 0.3)}" text-anchor="middle"
          style="fill:#e5e7eb;font-weight:600;font-size:${nodeLabelSize}px">${escapeHTML(label)}</text>
        ${badge}
      </g>`;
    }
  }

  svg.innerHTML = defs + `<g>${edges}</g><g>${nodes}</g><g>${biasNodes}</g>`;
  const tooltip = document.getElementById("edgeTooltip");

  svg.querySelectorAll(".edge-group").forEach((g) => {
    g.addEventListener("mousemove", (e) => {
      const weight = g.dataset.weight;

      tooltip.innerHTML = `
        <div style="
          font-size:12px;
          opacity:.7;
          margin-bottom:4px;
          text-transform:uppercase;
          letter-spacing:.8px;
        ">
          ${t("weight")}
        </div>

        <div style="
          font-size:24px;
          font-weight:700;
          line-height:1;
        ">
          ${weight}
        </div>
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

    if (value === undefined) return;

    node.addEventListener("mousemove", (e) => {
      tooltip.innerHTML = `<b>${t("activation")}</b>: ${Number(value).toFixed(4)}`;

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
}

// ========= Build Network =========
function buildNetwork() {
  console.log("BUILD ARCH", arch);

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
