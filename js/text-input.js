const MAX_TEXT_ALPHABET_SIZE = 63;

function uniqueCharacters(value) {
  return Array.from(new Set(Array.from(String(value || "").normalize("NFC"))));
}

function normalizeInputConfig(config, numericSize = inputSize || 1) {
  const mode = config?.mode === "text" ? "text" : "numeric";
  const rawAlphabet =
    config?.text?.alphabet || inputConfig?.text?.alphabet || "abcdefghijklmnopqrstuvwxyz ";
  const alphabet = uniqueCharacters(rawAlphabet)
    .slice(0, MAX_TEXT_ALPHABET_SIZE)
    .join("");

  return {
    mode,
    numericSize: Math.max(1, Number(config?.numericSize) || numericSize || 1),
    text: {
      alphabet: alphabet || "abcdefghijklmnopqrstuvwxyz ",
      lowercase: config?.text?.lowercase !== false,
      encoding: "frequency",
    },
  };
}

function serializeInputConfig() {
  return JSON.parse(JSON.stringify(inputConfig));
}

function applyInputConfig(config, numericSize = inputSize || 1) {
  inputConfig = normalizeInputConfig(config, numericSize);
  syncInputModeControls();
}

function textAlphabetCharacters() {
  return uniqueCharacters(inputConfig.text.alphabet);
}

function textFeatureCount() {
  return textAlphabetCharacters().length + 1;
}

function normalizeInputText(text) {
  const normalized = String(text || "").normalize("NFC").replace(/\s+/g, " ");
  return inputConfig.text.lowercase ? normalized.toLocaleLowerCase() : normalized;
}

function encodeTextInput(text) {
  const alphabet = textAlphabetCharacters();
  const indexByCharacter = new Map(alphabet.map((character, index) => [character, index]));
  const vector = Array(alphabet.length + 1).fill(0);
  const characters = Array.from(normalizeInputText(text));

  if (!characters.length) return vector;

  for (const character of characters) {
    const index = indexByCharacter.get(character);
    vector[index === undefined ? alphabet.length : index] += 1;
  }

  return vector.map((count) => count / characters.length);
}

function deriveTextAlphabet(texts) {
  const characters = new Set();

  for (const text of texts) {
    for (const character of Array.from(normalizeInputText(text))) {
      characters.add(character);
    }
  }

  return Array.from(characters)
    .sort((a, b) => a.localeCompare(b, getLang()))
    .slice(0, MAX_TEXT_ALPHABET_SIZE)
    .join("");
}

function updateTextDatasetEncoding() {
  if (inputConfig.mode !== "text" || !dataset.rawText?.length) return;
  dataset.X = dataset.rawText.map(encodeTextInput);
}

function updateTextFeatureCount() {
  const count = document.getElementById("textFeatureCount");
  if (!count) return;
  count.textContent = `${textFeatureCount()} ${t("textFeatures")}`;
}

function updateTextEncodingPreview(text, vector = encodeTextInput(text)) {
  const preview = document.getElementById("textEncodingPreview");
  if (!preview) return;

  const used = vector.filter((value) => value > 0).length;
  preview.textContent = t("textEncodingSummary")
    .replace("{used}", used)
    .replace("{total}", vector.length);
}

function syncInputModeControls() {
  const numeric = document.getElementById("inputModeNumeric");
  const text = document.getElementById("inputModeText");
  const settings = document.getElementById("textInputSettings");
  const alphabet = document.getElementById("textAlphabet");
  const lowercase = document.getElementById("textLowercase");

  if (numeric) numeric.checked = inputConfig.mode === "numeric";
  if (text) text.checked = inputConfig.mode === "text";
  settings?.classList.toggle("d-none", inputConfig.mode !== "text");
  if (alphabet) alphabet.value = inputConfig.text.alphabet;
  if (lowercase) lowercase.checked = inputConfig.text.lowercase;
  updateTextFeatureCount();
}

function resetDatasetForInputChange() {
  dataset = { X: [], y: [], rawText: [] };
  const preset = document.getElementById("presetDataset");
  if (preset) preset.value = "";
  const info = document.getElementById("csvInfo");
  if (info) {
    info.textContent = t("csvNoDataset");
    info.dataset.auto = "empty";
  }
}

function syncInputLayerToConfig({ rebuild = true } = {}) {
  const inputLayer = arch.find((layer) => layer.type === "input");
  const size =
    inputConfig.mode === "text" ? textFeatureCount() : inputConfig.numericSize;

  inputSize = size;
  if (inputLayer) inputLayer.neurons = size;

  if (rebuild) {
    buildNetwork();
    renderArchitecture();
    updateJSON();
  }
}

function setInputMode(mode) {
  const nextMode = mode === "text" ? "text" : "numeric";
  if (nextMode === inputConfig.mode) return;

  if (inputConfig.mode === "numeric") {
    inputConfig.numericSize = inputSize;
  }

  inputConfig.mode = nextMode;
  resetDatasetForInputChange();
  syncInputLayerToConfig();
  syncInputModeControls();
}

function wireInputModeControls() {
  document.getElementById("inputModeNumeric")?.addEventListener("change", (event) => {
    if (event.target.checked) setInputMode("numeric");
  });

  document.getElementById("inputModeText")?.addEventListener("change", (event) => {
    if (event.target.checked) setInputMode("text");
  });

  document.getElementById("textAlphabet")?.addEventListener("change", (event) => {
    const alphabet = uniqueCharacters(event.target.value)
      .slice(0, MAX_TEXT_ALPHABET_SIZE)
      .join("");
    inputConfig.text.alphabet = alphabet || "abcdefghijklmnopqrstuvwxyz ";
    event.target.value = inputConfig.text.alphabet;
    updateTextDatasetEncoding();
    syncInputLayerToConfig();
    updateTextFeatureCount();
  });

  document.getElementById("textLowercase")?.addEventListener("change", (event) => {
    inputConfig.text.lowercase = event.target.checked;
    if (dataset.rawText?.length) {
      const derivedAlphabet = deriveTextAlphabet(dataset.rawText);
      if (derivedAlphabet) inputConfig.text.alphabet = derivedAlphabet;
    }
    updateTextDatasetEncoding();
    syncInputLayerToConfig();
    syncInputModeControls();
  });

  syncInputModeControls();
}
