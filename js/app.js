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

  on("presetDataset", "change", (e) => {
    loadPreset(e.target.value);
  });

  on("btnTrain", "click", () => {
    if (dataset.X.length === 0) {
      alert(t("trainNoDataset"));
      return;
    }

    const tbtn = $("#btnTrain");
    const sbtn = $("#btnStop");

    if (tbtn) tbtn.disabled = true;
    if (sbtn) sbtn.disabled = false;

    trainLoop();
  });

  on("btnStop", "click", () => {
    stopFlag = true;
  });

  on("trainingModeStandard", "change", (e) => {
    if (e.target.checked) setTrainingMode("standard");
  });
  on("trainingModeStep", "change", (e) => {
    if (e.target.checked) setTrainingMode("step");
  });
  on("btnStepReset", "click", () => resetStepTraining());
  on("btnStepNext", "click", () => nextStepTraining());

  on("btnPredict", "click", () => {
    if (inputConfig.mode === "numeric") {
      const ti = document.querySelectorAll("#testInputs [data-ti]");
      if (ti.length !== inputSize) renderTestInputs();
    }
    predictOnce();
  });

  on("btnSaveNetwork", "click", async () => {
    await saveNetwork();

    await populateNetworksSelect();
  });

  on("btnUpdateNetwork", "click", async () => {
    if (!currentNetworkId) {
      alert("No network loaded");
      return;
    }

    await updateNetwork(currentNetworkId);

    await populateNetworksSelect();
  });

  on("btnAddHidden", "click", () => addLayer("hidden"));
  on("btnImportJSON", "click", () => {
    document.getElementById("jsonFile")?.click();
  });
  on("btnExport", "click", () => {
    const mode = document.getElementById("exportMode")?.value || "full";
    exportJSONFile(mode);
  });
  on("jsonCompact", "change", (e) => {
    jsonCompact = e.target.checked;
    updateJSON();
  });
  on("btnDownloadJSON", "click", () => downloadWeights());

  on("btnCopyJSON", "click", async () => {
    await navigator.clipboard.writeText(
      document.getElementById("jsonArea")?.value || "",
    );
    const b = document.getElementById("btnCopyJSON");
    if (!b) return;
    const txt = b.innerHTML;
    b.innerHTML = t("copied");
    setTimeout(() => (b.innerHTML = txt), 1200);
  });

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

  const jsonFile = document.getElementById("jsonFile");
  if (jsonFile && !jsonFile._wiredJson) {
    jsonFile._wiredJson = true;
    jsonFile.addEventListener("change", (e) => {
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
    "btnStepNext",
    "btnPredict",
    "btnAddHidden",
    "btnExport",
    "btnDownloadJSON",
    "btnCopyJSON",
    "btnLangToggle",
    "btnLogout",
    "csvFile",
    "csvInfoBtn",
  ];

  console.group("%c[NeuroBuilder] Check UI", "color:#0ea5e9;font-weight:700");
  ids.forEach((id) => {
    const el = document.getElementById(id);
    console[el ? "log" : "warn"](
      `${el ? "✓" : "✗"} ${id} ${el ? "found" : "missing"}`,
    );
  });
  console.groupEnd();
}

// ========= Init =========
document.addEventListener("DOMContentLoaded", async () => {
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
  let networkResizeTimer;
  window.addEventListener("resize", () => {
    clearTimeout(networkResizeTimer);
    networkResizeTimer = setTimeout(() => renderNNVis(), 100);
  });

  arch = [
    {
      id: crypto.randomUUID(),
      type: "input",
      neurons: 2,
    },
    {
      id: crypto.randomUUID(),
      type: "hidden",
      neurons: 4,
      activation: "relu",
      bias: true,
    },
    {
      id: crypto.randomUUID(),
      type: "output",
      neurons: 1,
      activation: "sigmoid",
      bias: true,
    },
  ];
  inputSize = 2;
  outputSize = 1;
  buildNetwork();
  renderArchitecture();

  bindUIControls();
  updateStepTrainingControls();
  wireInputModeControls();
  initTutorialControls();
  wireCsvInputs();
  attachPopoverGlobalClosers();
  applyI18n();
  initCsvInfoSafe();
  await initAuthUI();
  updateNetworkTitle();
});
