// ========= JSON Export/Import & sync =========
function updateJSON() {
  const j = {
    architecture: {
      layers: arch,
    },

    weights: net.layers.map((l) => ({
      in: l.in,
      out: l.out,
      activation: l.activation,
      useBias: l.useBias,
      W: l.W,
      b: l.b,
    })),
  };

  const jsonArea = $("#jsonArea");
  if (!jsonArea) return;

  jsonArea.value = jsonCompact ? JSON.stringify(j) : JSON.stringify(j, null, 2);
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

        buildNetwork();

        for (let i = 0; i < net.layers.length; i++) {
          net.layers[i].W = o.layers[i].W;
          net.layers[i].b = o.layers[i].b;
        }

        renderArchitecture();
        renderTestInputs();
        renderNNVis();
        updateJSON();

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

        renderArchitecture();
        renderTestInputs();
        renderNNVis();
        updateJSON();

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
  let j;
  let filename;

  if (mode === "architecture") {
    j = {
      architecture: {
        layers: arch.map((l) => ({
          type: l.type,
          neurons: l.neurons,
          activation: l.activation,
          bias: l.bias,
        })),
      },
    };

    filename = "neurobuilder-architecture.json";
  } else if (mode === "weights") {
    j = {
      layers: net.layers.map((l) => ({
        in: l.in,
        out: l.out,
        activation: l.activation,
        useBias: l.useBias,
        W: l.W,
        b: l.b,
      })),
    };

    filename = "neurobuilder-weights.json";
  } else {
    j = {
      architecture: {
        layers: arch.map((l) => ({
          type: l.type,
          neurons: l.neurons,
          activation: l.activation,
          bias: l.bias,
        })),
      },

      weights: net.layers.map((l) => ({
        in: l.in,
        out: l.out,
        activation: l.activation,
        useBias: l.useBias,
        W: l.W,
        b: l.b,
      })),

      dataset: {
        X: dataset.X,
        y: dataset.y,
      },
    };

    filename = "neurobuilder-full-network.json";
  }

  const blob = new Blob([JSON.stringify(j, null, 2)], {
    type: "application/json",
  });

  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = filename;
  a.click();
  URL.revokeObjectURL(a.href);
}

function exportArchitecture() {
  const j = {
    architecture: {
      layers: arch.map((l) => ({
        type: l.type,
        neurons: l.neurons,
        activation: l.activation,
        bias: l.bias,
      })),
    },
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

