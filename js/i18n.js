// ========= Dizionario UI =========
const I18N = {
  it: {
    input: "Input",
    output: "Output",
    hiddenLayer: "Layer nascosto",
    neurons: "Neuroni",
    inputSize: "Dimensione input",
    activation: "Attivazione",
    bias: "Bias",
    network: "Nuova rete",
    langButton: "EN",
    emptyArchitecture: "Trascina qui i layer dalla palette…",
    remove: "Rimuovi",
    weight: "Peso",
    csvNoFile: "Nessun file selezionato.",
    csvEmpty: "Il file sembra vuoto.",
    csvNeedCols: "Servono almeno 2 colonne (feature + target).",
    csvNonNumeric: "Valori non numerici alla riga",
    csvLoaded: "caricato",
    csvExamples: "esempi",
    csvFeatures: "feature",
    csvDelimiter: "delimitatore",
    csvNoDataset: "Nessun dataset caricato",
    csvReadError: "Impossibile leggere il file CSV.",
    csvParseError: "Errore CSV: ",
    presetXorLoaded: "Caricato preset XOR (4 esempi)",
    presetLinearLoaded: "Caricato dataset lineare (200 esempi)",
    trainNoDataset: "Carica o scegli un preset/CSV prima di allenare.",
    inputMismatch: "Dimensione input non coerente con la rete.",
    importWeightsOk: "✅ Import riuscito (formato pesi).",
    importArchOk: "✅ Import riuscito (architettura",
    importWeightsSuffix: " + pesi",
    importClose: ").",
    jsonInvalid: "❌ JSON non valido: ",
    jsonUnknown:
      "Formato non riconosciuto. Attesi: {layers:[...]} oppure {architecture:[...], weights?:[...]}",
    layersEmpty: "layers vuoto",
    architectureMissing: 'manca "architecture"',
    copied: '<i class="bi bi-clipboard-check"></i> Copiato!',
    csvPopoverTitle: "Formato CSV richiesto",
    csvPopoverHtml: `
      <div>
        <b>• Senza intestazioni</b><br>
        • Separatore: virgola (<code>,</code>)<br>
        • Tutto numerico (niente NaN)<br>
        • <b>Ultima colonna = target</b> (0/1)<br>
        • Esempio:<br>
        <code>0,0,0<br>0,1,1<br>1,0,1<br>1,1,0</code>
      </div>`,
  },
  en: {
    input: "Input",
    output: "Output",
    hiddenLayer: "Hidden layer",
    neurons: "Neurons",
    inputSize: "Input size",
    activation: "Activation",
    bias: "Bias",
    network: "New network",
    langButton: "IT",
    emptyArchitecture: "Drag layers here from the palette…",
    remove: "Remove",
    weight: "Weight",
    csvNoFile: "No file selected.",
    csvEmpty: "The file seems empty.",
    csvNeedCols: "At least 2 columns are required (features + target).",
    csvNonNumeric: "Non-numeric values at row",
    csvLoaded: "loaded",
    csvExamples: "examples",
    csvFeatures: "features",
    csvDelimiter: "delimiter",
    csvNoDataset: "No dataset loaded",
    csvReadError: "Can't read CSV file.",
    csvParseError: "CSV error: ",
    presetXorLoaded: "XOR preset loaded (4 examples)",
    presetLinearLoaded: "Linear dataset loaded (200 examples)",
    trainNoDataset: "Load or choose a preset/CSV before training.",
    inputMismatch: "Input size doesn't match the network.",
    importWeightsOk: "✅ Import OK (weight format).",
    importArchOk: "✅ Import OK (Architecture",
    importWeightsSuffix: " + weights",
    importClose: ").",
    jsonInvalid: "❌ JSON not valid: ",
    jsonUnknown:
      "Format unknown. Expected: {layers:[...]} or {architecture:[...], weights?:[...]}",
    layersEmpty: "layers empty",
    architectureMissing: '"Architecture" missing',
    copied: '<i class="bi bi-clipboard-check"></i> Copied!',
    csvPopoverTitle: "Required CSV format",
    csvPopoverHtml: `
      <div>
        <b>• No headers</b><br>
        • Separator: comma (<code>,</code>)<br>
        • All numeric values (no NaN)<br>
        • <b>Last column = target</b> (0/1)<br>
        • Example:<br>
        <code>0,0,0<br>0,1,1<br>1,0,1<br>1,1,0</code>
      </div>`,
  },
};

function applyI18n() {
  const lang = getLang();

  $$("[data-i18n]").forEach((el) => {
    const key = el.dataset.i18n;

    if (I18N_HTML[lang]?.[key] !== undefined) {
      if (el.tagName === "OPTION") {
        el.textContent = I18N_HTML[lang][key];
      } else {
        el.innerHTML = I18N_HTML[lang][key];
      }
    }
  });

  $$("[data-i18n-placeholder]").forEach((el) => {
    const key = el.dataset.i18nPlaceholder;

    if (I18N_HTML[lang]?.[key] !== undefined) {
      el.placeholder = I18N_HTML[lang][key];
    }
  });

  $$("[data-i18n-aria]").forEach((el) => {
    const key = el.dataset.i18nAria;

    if (I18N_HTML[lang]?.[key] !== undefined) {
      el.setAttribute("aria-label", I18N_HTML[lang][key]);
    }
  });

  const btn = document.getElementById("btnLangToggle");
  if (btn) {
    btn.textContent = t("langButton");
  }

  const csvInfo = document.getElementById("csvInfo");

  if (csvInfo && (!dataset.X.length || csvInfo.dataset.auto === "empty")) {
    csvInfo.textContent = t("csvNoDataset");
    csvInfo.dataset.auto = "empty";
  }

  initCsvInfoSafe();
}

const I18N_HTML = {
  it: {
    palette: "Palette",
    dragHere: "Trascina i blocchi qui sotto 👉",
    input: "Input",
    output: "Output",
    hiddenLayer: "Layer nascosto",
    neurons: "Neuroni",
    inputSize: "Dimensione input",
    activation: "Attivazione",
    bias: "Bias",
    legendPositive: "Verde = peso positivo",
    legendNegative: "Rosso = peso negativo",
    legendThickness: "Spessore = intensità del peso",
    legendHover: "Passa col mouse sui collegamenti per vedere il peso numerico",
    legendNodeColor: "Colore nodo = livello di attivazione del neurone",
    clear: "Pulisci",
    architecture: "Architettura",
    training: "Training",
    learningRate: "Learning rate",
    epochs: "Epoche",
    batch: "Batch",
    train: "Allena",
    stop: "Stop",
    quickStart: "Quick Start (XOR)",
    loss: "Loss",
    accuracy: "Accuratezza",
    network: "Visualizzazione rete",
    prediction: "Test predizione",
    predict: "Predici",
    outputLabel: "Output",
    dataset: "Dataset",
    load: "Carica",
    csvDropText: "Trascina CSV qui",
    noDataset: "Nessun dataset caricato",
    json: "JSON",
    export: "Export",
    download: "Download",
    copy: "Copy",

    linearPreset: "Lineare (x + y > 1)",
  },

  en: {
    palette: "Palette",
    dragHere: "Drag the blocks below 👉",

    input: "Input",
    output: "Output",
    hiddenLayer: "Hidden layer",
    neurons: "Neurons",
    inputSize: "Input size",
    activation: "Activation",
    bias: "Bias",
    legendPositive: "Green = positive weight",
    legendNegative: "Red = negative weight",
    legendThickness: "Thickness = weight strength",
    legendHover: "Hover over the connections to see the numeric weight",
    legendNodeColor: "Node color = neuron activation level",
    clear: "Clear",
    architecture: "Architecture",
    training: "Training",
    learningRate: "Learning rate",
    epochs: "Epochs",
    batch: "Batch",
    train: "Train",
    stop: "Stop",
    quickStart: "Quick Start (XOR)",
    loss: "Loss",
    accuracy: "Accuracy",
    network: "Network visualization",
    prediction: "Prediction Test",
    predict: "Predict",
    outputLabel: "Output",
    dataset: "Dataset",
    load: "Load",
    csvDropText: "Drop CSV here",
    noDataset: "No dataset loaded",
    json: "JSON",
    export: "Export",
    download: "Download",
    copy: "Copy",

    linearPreset: "Linear (x + y > 1)",
  },
};
