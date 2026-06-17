// ========= Dataset Preset & CSV =========
function loadPreset(name) {
  if (name === "xor") {
    dataset.X = [
      [0, 0],
      [0, 1],
      [1, 0],
      [1, 1],
    ];
    dataset.y = [[0], [1], [1], [0]];
    inputSize = 2;
    outputSize = 1;
    ensureIOInArch();
    const csvInfo = $("#csvInfo");
    if (csvInfo) {
      csvInfo.textContent = t("presetXorLoaded");
      csvInfo.dataset.auto = "loaded";
    }
  } else if (name === "linsep") {
    const X = [];
    const y = [];
    const r = rng(7);
    for (let i = 0; i < 200; i++) {
      const a = r();
      const b = r();
      X.push([a, b]);
      y.push([a + b > 1 ? 1 : 0]);
    }
    dataset.X = X;
    dataset.y = y;
    inputSize = 2;
    outputSize = 1;
    ensureIOInArch();
    const csvInfo = $("#csvInfo");
    if (csvInfo) {
      csvInfo.textContent = t("presetLinearLoaded");
      csvInfo.dataset.auto = "loaded";
    }
  }
  renderTestInputs();
}

function ensureIOInArch() {
  let inL = arch.find((l) => l.type === "input");

  if (!inL) {
    arch.unshift({
      id: crypto.randomUUID(),
      type: "input",
      neurons: inputSize,
    });
  } else {
    inL.neurons = inputSize;
  }

  let outL = arch.find((l) => l.type === "output");

  if (!outL) {
    arch.push({
      id: crypto.randomUUID(),
      type: "output",
      neurons: outputSize,
      activation: "sigmoid",
      bias: true,
    });
  } else {
    outL.neurons = outputSize;
  }

  buildNetwork();
  renderArchitecture();
  updateJSON();
}

function handleCSVFile(file) {
  if (!file) {
    alert(t("csvNoFile"));
    return;
  }

  console.log("[CSV] reading file:", file.name, file.type, file.size, "bytes");

  const infoEl = document.getElementById("csvInfo");
  const setInfo = (msg, mode = "loaded") => {
    if (infoEl) {
      infoEl.textContent = msg;
      infoEl.dataset.auto = mode;
    }
  };

  const sniffDelimiter = (text) => {
    const counts = {
      ",": (text.match(/,/g) || []).length,
      ";": (text.match(/;/g) || []).length,
      "\t": (text.match(/\t/g) || []).length,
    };
    return Object.entries(counts).sort((a, b) => b[1] - a[1])[0][0] || ",";
  };

  const toNumber = (token) => {
    const value = token.trim().replace(",", ".");
    const n = Number(value);
    return Number.isFinite(n) ? n : NaN;
  };

  const parse = (text) => {
    let lines = text
      .split(/\r?\n/)
      .map((line) => line.trim())
      .filter(Boolean);

    if (lines.length === 0) throw new Error(t("csvEmpty"));

    const delim = sniffDelimiter(lines.slice(0, 50).join("\n"));
    if (/[a-zA-Z]/.test(lines[0])) lines.shift();

    const rows = lines.map((line) => line.split(delim));
    const cols = rows[0].length;
    if (cols < 2) throw new Error(t("csvNeedCols"));

    const data = rows.map((row, ri) => {
      const nums = row.map(toNumber);
      if (nums.some((x) => Number.isNaN(x))) {
        throw new Error(`${t("csvNonNumeric")} ${ri + 1}.`);
      }
      return nums;
    });

    const inputSizeNew = cols - 1;
    const X = data.map((row) => row.slice(0, inputSizeNew));
    const y = data.map((row) => [row[cols - 1]]);
    return { X, y, inputSizeNew, delimiter: delim };
  };

  const fr = new FileReader();

  fr.onload = () => {
    try {
      const text = fr.result;
      const { X, y, inputSizeNew, delimiter } = parse(text);

      dataset.X = X;
      dataset.y = y;
      inputSize = inputSizeNew;
      outputSize = 1;

      ensureIOInArch();
      renderTestInputs();

      setInfo(
        `CSV "${file.name}" ${t("csvLoaded")}: ${X.length} ${t(
          "csvExamples",
        )}, ${inputSize} ${t("csvFeatures")} (${t("csvDelimiter")} "${
          delimiter === "\t" ? "TAB" : delimiter
        }")`,
      );

      console.log("[CSV] OK. First rows:", X.slice(0, 3), y.slice(0, 3));
    } catch (err) {
      console.error("[CSV] Parsing error:", err);
      setInfo(t("csvNoDataset"), "empty");
      alert("❌ " + t("csvParseError") + err.message);
    }
  };

  fr.onerror = () => {
    console.error("[CSV] FileReader error:", fr.error);
    setInfo(t("csvNoDataset"), "empty");
    alert("❌ " + t("csvReadError"));
  };

  fr.readAsText(file);
}

function wireCsvInputs() {
  const fi = document.getElementById("csvFile");
  if (fi) {
    if (!fi._wiredCsv) {
      const clone = fi.cloneNode(true);
      fi.parentNode.replaceChild(clone, fi);
      clone._wiredCsv = true;
      clone.addEventListener("change", (e) => {
        const f = e.target.files && e.target.files[0];
        console.log("[CSV] change →", f?.name || "(no file)");
        handleCSVFile(f);
      });
    }
  } else {
    console.warn("[CSV] #csvFile not found");
  }

  const dz = document.getElementById("csvDrop");
  if (dz && !dz._wiredDrop) {
    dz._wiredDrop = true;
    dz.addEventListener("dragover", (e) => {
      e.preventDefault();
      dz.classList.add("dragover");
    });
    dz.addEventListener("dragleave", () => dz.classList.remove("dragover"));
    dz.addEventListener("drop", (e) => {
      e.preventDefault();
      dz.classList.remove("dragover");
      const file = e.dataTransfer?.files?.[0];
      handleCSVFile(file);
    });
  }
}

function observeCsvInputs() {
  const obs = new MutationObserver(() => {
    const fi = document.getElementById("csvFile");
    if (fi && !fi._wiredCsv) wireCsvInputs();
  });
  obs.observe(document.documentElement, { childList: true, subtree: true });
}

function attachCsvDnD() {
  const dz = $("#csvDrop");
  if (!dz) return;

  const clone = dz.cloneNode(true);
  dz.parentNode.replaceChild(clone, dz);
  const el = $("#csvDrop");

  el.addEventListener("dragover", (e) => {
    e.preventDefault();
    el.classList.add("dragover");
  });

  el.addEventListener("dragleave", () => el.classList.remove("dragover"));

  el.addEventListener("drop", (e) => {
    e.preventDefault();
    el.classList.remove("dragover");
    handleCSVFile(e.dataTransfer.files?.[0]);
  });
}

