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
      maintainAspectRatio: false,
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
  resetStepTraining({ render: false });
  const lr = Number(document.getElementById("lr")?.value ?? 0.1);
  const epochs = Number(document.getElementById("epochs")?.value ?? 50);
  const rand = rng(42);
  if (!dataset.X.length || !dataset.y.length) {
    console.warn("No dataset loaded");
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
      dataset.X.length,
    );
    const batches = getBatches(X, y, batchSize, rand);

    for (let batchIndex = 0; batchIndex < batches.length; batchIndex++) {
      const b = batches[batchIndex];
      const ypred = net.forward(b.X);
      const [, dLdy] = bce(ypred, b.y);
      net.backward(dLdy, lrNow);

      step++;
      if (step % VIS_EVERY_STEPS === 0 && b.X.length > 0) {
        computeNodeColorsForInput(b.X[0]);
        renderNNVis();
        await new Promise((resolve) => setTimeout(resolve, 0));
      }

      updateTrainingProgress(ep, epochs, batchIndex + 1, batches.length);
      if (stopFlag) break;
    }

    const fullPred = net.forward(X);
    const [L] = bce(fullPred, y);
    const acc = accuracyBinary(fullPred, y);

    console.log(
      `Epoch ${ep} → Loss: ${L.toFixed(4)} | Acc: ${(acc * 100).toFixed(
        1,
      )}% | LR: ${lrNow.toFixed(5)}`,
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

  $("#epochProgressBar").style.width = "100%";
  $("#batchProgressBar").style.width = "100%";
  $("#epochProgressText").textContent = "Completed";
  $("#batchProgressText").textContent = "Completed";

  updateJSON();
}

function updateTrainingProgress(epoch, totalEpochs, batch, totalBatches) {
  const epochText = $("#epochProgressText");
  const epochBar = $("#epochProgressBar");

  const batchText = $("#batchProgressText");
  const batchBar = $("#batchProgressBar");

  if (epochText) epochText.textContent = `Epoch ${epoch}/${totalEpochs}`;

  if (epochBar) epochBar.style.width = (epoch / totalEpochs) * 100 + "%";

  if (batchText) batchText.textContent = `Batch ${batch}/${totalBatches}`;

  if (batchBar) batchBar.style.width = (batch / totalBatches) * 100 + "%";
}

// ========= Predict =========
function renderPredictionOutputs(values = []) {
  const container = document.getElementById("predictOutputs");
  if (!container) return;

  const count = Math.max(outputSize, values.length);
  container.innerHTML = "";

  for (let i = 0; i < count; i++) {
    const value = values[i];
    const formattedValue = Number.isFinite(value)
      ? Number(value.toFixed(5)).toString()
      : "-";
    const roundedValue = Number.isFinite(value) ? Math.round(value) : "-";
    const box = document.createElement("div");
    box.className = "prediction-output-box";
    box.innerHTML = `
      <span class="prediction-output-label">y${i + 1}</span>
      <output class="prediction-output-value">${formattedValue}</output>
      <output class="prediction-output-rounded" title="${t("roundedOutput")}">${roundedValue}</output>
    `;
    container.appendChild(box);
  }
}

function predictOnce() {
  const textValue = document.getElementById("textPredictInput")?.value || "";
  const vals =
    inputConfig.mode === "text"
      ? encodeTextInput(textValue)
      : $$("#testInputs [data-ti]").map((i) => Number(i.value));

  if (inputConfig.mode === "text" && !textValue.trim()) {
    alert(t("textEmpty"));
    return;
  }

  if (vals.length !== inputSize) {
    alert(t("inputMismatch"));
    return;
  }

  if (!net.layers.length) {
    alert("Rete non configurata");
    return;
  }

  const out = net.forward([vals]);

  if (inputConfig.mode === "text") {
    updateTextEncodingPreview(textValue, vals);
  }

  computeNodeColorsForInput(vals);

  renderPredictionOutputs(out[0]);

  renderNNVis();
}

function updateNetworkTitle() {
  const title = document.getElementById("networkTitle");

  if (!title) return;

  if (currentNetworkName) {
    title.innerHTML = `<i class="bi bi-eye"></i> ${currentNetworkName}`;
  } else {
    title.innerHTML = `<i class="bi bi-eye"></i> ${t("network")}`;
  }
}
