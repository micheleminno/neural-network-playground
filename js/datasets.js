// ========= Dataset Preset & CSV =========
const ENGLISH_SENTIMENT_PRESET = [
  ["I loved this lesson", 1],
  ["This was clear and helpful", 1],
  ["The explanation was excellent", 1],
  ["I really enjoyed the activity", 1],
  ["This is a great example", 1],
  ["The course is useful and engaging", 1],
  ["Everything was easy to understand", 1],
  ["I am happy with the result", 1],
  ["The exercise was fun", 1],
  ["This approach works very well", 1],
  ["The instructions were simple and precise", 1],
  ["I learned a lot today", 1],
  ["The teacher did a wonderful job", 1],
  ["This tool is intuitive", 1],
  ["The answer is correct", 1],
  ["What a brilliant idea", 1],
  ["The experience was positive", 1],
  ["I would recommend this course", 1],
  ["The example made the topic easier", 1],
  ["I feel confident now", 1],
  ["I hated this lesson", 0],
  ["This was confusing and unhelpful", 0],
  ["The explanation was terrible", 0],
  ["I did not enjoy the activity", 0],
  ["This is a bad example", 0],
  ["The course is boring and frustrating", 0],
  ["Everything was difficult to understand", 0],
  ["I am unhappy with the result", 0],
  ["The exercise was annoying", 0],
  ["This approach does not work", 0],
  ["The instructions were vague and incorrect", 0],
  ["I learned nothing today", 0],
  ["The teacher did a poor job", 0],
  ["This tool is complicated", 0],
  ["The answer is wrong", 0],
  ["What an awful idea", 0],
  ["The experience was negative", 0],
  ["I would not recommend this course", 0],
  ["The example made the topic harder", 0],
  ["I still feel completely lost", 0],
];

function loadPreset(name) {
  if (name === "xor") {
    applyInputConfig({ ...serializeInputConfig(), mode: "numeric", numericSize: 2 }, 2);
    dataset.X = [
      [0, 0],
      [0, 1],
      [1, 0],
      [1, 1],
    ];
    dataset.y = [[0], [1], [1], [0]];
    dataset.rawText = [];
    inputSize = 2;
    outputSize = 1;
    ensureIOInArch();
    const csvInfo = $("#csvInfo");
    if (csvInfo) {
      csvInfo.textContent = t("presetXorLoaded");
      csvInfo.dataset.auto = "loaded";
    }
  } else if (name === "linsep") {
    applyInputConfig({ ...serializeInputConfig(), mode: "numeric", numericSize: 2 }, 2);
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
    dataset.rawText = [];
    inputSize = 2;
    outputSize = 1;
    ensureIOInArch();
    const csvInfo = $("#csvInfo");
    if (csvInfo) {
      csvInfo.textContent = t("presetLinearLoaded");
      csvInfo.dataset.auto = "loaded";
    }
  } else if (name === "sentiment-en") {
    applyInputConfig(
      {
        ...serializeInputConfig(),
        mode: "text",
        text: {
          ...serializeInputConfig().text,
          alphabet: "abcdefghijklmnopqrstuvwxyz ",
          lowercase: true,
        },
      },
      inputSize,
    );

    const rawText = ENGLISH_SENTIMENT_PRESET.map(([text]) => text);
    inputConfig.text.alphabet = deriveTextAlphabet(rawText);
    dataset.rawText = rawText;
    dataset.X = rawText.map(encodeTextInput);
    dataset.y = ENGLISH_SENTIMENT_PRESET.map(([, label]) => [label]);
    inputSize = textFeatureCount();
    outputSize = 1;
    ensureIOInArch();
    syncInputModeControls();

    const csvInfo = $("#csvInfo");
    if (csvInfo) {
      csvInfo.textContent = t("presetSentimentLoaded");
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

  const toNumber = (token) => {
    const value = String(token).trim().replace(",", ".");
    const n = Number(value);
    return Number.isFinite(n) ? n : NaN;
  };

  const parse = (text) => {
    if (!window.Papa) throw new Error(t("csvParserUnavailable"));

    const parsed = Papa.parse(text, {
      header: false,
      dynamicTyping: false,
      skipEmptyLines: "greedy",
    });

    const fatalError = parsed.errors.find(
      (error) => error.code !== "TooFewFields" && error.code !== "TooManyFields",
    );
    if (fatalError) throw new Error(fatalError.message);

    let rows = parsed.data.map((row) => row.map((cell) => String(cell).trim()));
    if (!rows.length) throw new Error(t("csvEmpty"));

    const outputColumns =
      arch.find((layer) => layer.type === "output")?.neurons || outputSize;
    const firstRow = rows[0];
    const firstTextCell = firstRow[0]?.toLocaleLowerCase() || "";
    const hasHeader =
      inputConfig.mode === "text"
        ? ["text", "testo", "sentence", "phrase", "input"].includes(firstTextCell) ||
          firstRow.slice(1).some((cell) => Number.isNaN(toNumber(cell)))
        : firstRow.some((cell) => Number.isNaN(toNumber(cell)));

    if (hasHeader) rows = rows.slice(1);
    if (!rows.length) throw new Error(t("csvEmpty"));

    const cols = rows[0].length;
    if (cols < 2) throw new Error(t("csvNeedCols"));

    for (let rowIndex = 0; rowIndex < rows.length; rowIndex++) {
      if (rows[rowIndex].length !== cols) {
        throw new Error(
          t("csvRowColumnMismatch")
            .replace("{row}", rowIndex + 1)
            .replace("{expected}", cols)
            .replace("{actual}", rows[rowIndex].length),
        );
      }
    }

    if (inputConfig.mode === "text") {
      const expectedCols = 1 + outputColumns;
      if (cols !== expectedCols) {
        throw new Error(
          t("textCsvColumnMismatch")
            .replace("{expected}", expectedCols)
            .replace("{outputs}", outputColumns)
            .replace("{actual}", cols),
        );
      }

      const rawText = [];
      const y = [];

      rows.forEach((row, rowIndex) => {
        if (!row[0]) {
          throw new Error(`${t("textCsvEmptyText")} ${rowIndex + 1}.`);
        }
        const targets = row.slice(1).map(toNumber);
        if (targets.some(Number.isNaN)) {
          throw new Error(`${t("csvNonNumeric")} ${rowIndex + 1}.`);
        }
        rawText.push(row[0]);
        y.push(targets);
      });

      const derivedAlphabet = deriveTextAlphabet(rawText);
      if (derivedAlphabet) inputConfig.text.alphabet = derivedAlphabet;
      const X = rawText.map(encodeTextInput);
      return {
        X,
        y,
        rawText,
        inputSizeNew: textFeatureCount(),
        delimiter: parsed.meta.delimiter,
      };
    }

    const expectedInputSize =
      arch.find((layer) => layer.type === "input")?.neurons || inputSize;
    const expectedOutputSize =
      arch.find((layer) => layer.type === "output")?.neurons || outputSize;
    const expectedCols = expectedInputSize + expectedOutputSize;

    if (cols !== expectedCols) {
      throw new Error(
        t("csvColumnMismatch")
          .replace("{expected}", expectedCols)
          .replace("{inputs}", expectedInputSize)
          .replace("{outputs}", expectedOutputSize)
          .replace("{actual}", cols),
      );
    }

    const data = rows.map((row, ri) => {
      const nums = row.map(toNumber);
      if (nums.some((x) => Number.isNaN(x))) {
        throw new Error(`${t("csvNonNumeric")} ${ri + 1}.`);
      }
      return nums;
    });

    const X = data.map((row) => row.slice(0, expectedInputSize));
    const y = data.map((row) => row.slice(expectedInputSize));
    return {
      X,
      y,
      rawText: [],
      inputSizeNew: expectedInputSize,
      delimiter: parsed.meta.delimiter,
    };
  };

  const fr = new FileReader();

  fr.onload = () => {
    try {
      const text = fr.result;
      const { X, y, rawText, inputSizeNew, delimiter } = parse(text);

      dataset.X = X;
      dataset.y = y;
      dataset.rawText = rawText;
      resetStepTraining({ render: false });

      if (inputConfig.mode === "text") {
        inputSize = inputSizeNew;
        syncInputLayerToConfig({ rebuild: false });
        ensureIOInArch();
        syncInputModeControls();
      }

      renderTestInputs();

      setInfo(
        `CSV "${file.name}" ${t("csvLoaded")}: ${X.length} ${t(
          "csvExamples",
        )}, ${inputSize} ${
          inputConfig.mode === "text" ? t("textFeatures") : t("csvFeatures")
        } (${t("csvDelimiter")} "${
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
