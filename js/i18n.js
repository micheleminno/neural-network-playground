// ========= Dizionario UI =========
const I18N = {
  it: {
    input: "Input",
    output: "Output",
    hiddenLayer: "Layer nascosto",
    neurons: "Neuroni",
    inputSize: "Dimensione input",
    textFeatures: "feature testuali",
    textToPredict: "Testo",
    textPredictPlaceholder: "Scrivi il testo da classificare...",
    textEncodingSummary: "{used} di {total} feature attive",
    textEmpty: "Inserisci un testo da classificare.",
    activation: "Attivazione",
    weightedSum: "Somma pesata",
    inputLabel: "Input",
    valueLabel: "Valore",
    bias: "Bias",
    network: "Nuova rete",
    langButton: "EN",
    emptyArchitecture: "Trascina qui i layer dalla palette…",
    remove: "Rimuovi",
    weight: "Peso",
    csvNoFile: "Nessun file selezionato.",
    csvEmpty: "Il file sembra vuoto.",
    csvNeedCols: "Servono almeno 2 colonne (feature + target).",
    csvColumnMismatch:
      "La rete richiede {expected} colonne: {inputs} input e {outputs} output, ma il CSV ne contiene {actual}.",
    csvRowColumnMismatch:
      "La riga {row} contiene {actual} colonne invece di {expected}.",
    csvNonNumeric: "Valori non numerici alla riga",
    csvLoaded: "caricato",
    csvExamples: "esempi",
    csvFeatures: "feature",
    csvDelimiter: "delimitatore",
    csvNoDataset: "Nessun dataset caricato",
    csvReadError: "Impossibile leggere il file CSV.",
    csvParseError: "Errore CSV: ",
    csvParserUnavailable: "Il parser CSV non è disponibile.",
    textCsvColumnMismatch:
      "In modalità testo servono {expected} colonne: testo + {outputs} output, ma il CSV ne contiene {actual}.",
    textCsvEmptyText: "Testo vuoto alla riga",
    textConfigMismatch:
      "L'alfabeto importato non corrisponde alla dimensione input della rete.",
    presetXorLoaded: "Caricato preset XOR (4 esempi)",
    presetLinearLoaded: "Caricato dataset lineare (200 esempi)",
    presetSentimentLoaded:
      "Caricato sentiment inglese (40 frasi: 0 negativo, 1 positivo)",
    datasetPreviewCount: "Prime {shown} di {total} righe",
    chartEpochs: "Epoche",
    lossHelp:
      "La loss misura quanto le predizioni differiscono dagli output desiderati. Più è bassa, meglio la rete sta imparando.",
    accuracyHelp:
      "L'accuracy è la percentuale di esempi classificati correttamente. Più è alta, migliore è il risultato.",
    jsonItem: "elemento",
    jsonItems: "elementi",
    jsonProperty: "proprietà",
    jsonProperties: "proprietà",
    jsonExpandNode: "Espandi elemento",
    jsonCollapseNode: "Comprimi elemento",
    trainNoDataset: "Carica o scegli un preset/CSV prima di allenare.",
    standardTraining: "Normale",
    stepTraining: "Step-by-step",
    stepReset: "Reset",
    stepLoadExample: "Carica esempio",
    stepNext: "Passo successivo",
    stepReady: "Pronto per iniziare",
    stepExampleLoaded: "Nuovo esempio caricato",
    stepForwardLayer: "Forward pass · layer {layer}",
    stepLossCalculated: "Loss calcolata",
    stepBackwardLayer: "Backpropagation · layer {layer} aggiornato",
    stepUpdateComplete: "Aggiornamento completato · nuovo output calcolato",
    stepExampleCounter: "Esempio",
    stepExample: "Esempio",
    stepDesired: "Output desiderato",
    stepCurrent: "Output attuale",
    stepTargetShort: "target",
    stepNoNetwork: "Configura almeno un layer di output.",
    inputMismatch: "Dimensione input non coerente con la rete.",
    roundedOutput: "Valore arrotondato",
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
        <b>Numerico</b><br>
        • Input prima, output dopo<br>
        • Intestazione facoltativa<br>
        • Esempio: <code>0,1,1</code><br><br>
        <b>Testo</b><br>
        • Prima colonna = testo, poi gli output<br>
        • Intestazione facoltativa<br>
        • Esempio: <code>text,label<br>"lezione utile",1</code>
      </div>`,
  },
  en: {
    input: "Input",
    output: "Output",
    hiddenLayer: "Hidden layer",
    neurons: "Neurons",
    inputSize: "Input size",
    textFeatures: "text features",
    textToPredict: "Text",
    textPredictPlaceholder: "Type the text to classify...",
    textEncodingSummary: "{used} of {total} active features",
    textEmpty: "Enter some text to classify.",
    activation: "Activation",
    weightedSum: "Weighted sum",
    inputLabel: "Input",
    valueLabel: "Value",
    bias: "Bias",
    network: "New network",
    langButton: "IT",
    emptyArchitecture: "Drag layers here from the palette…",
    remove: "Remove",
    weight: "Weight",
    csvNoFile: "No file selected.",
    csvEmpty: "The file seems empty.",
    csvNeedCols: "At least 2 columns are required (features + target).",
    csvColumnMismatch:
      "The network requires {expected} columns: {inputs} inputs and {outputs} outputs, but the CSV contains {actual}.",
    csvRowColumnMismatch:
      "Row {row} contains {actual} columns instead of {expected}.",
    csvNonNumeric: "Non-numeric values at row",
    csvLoaded: "loaded",
    csvExamples: "examples",
    csvFeatures: "features",
    csvDelimiter: "delimiter",
    csvNoDataset: "No dataset loaded",
    csvReadError: "Can't read CSV file.",
    csvParseError: "CSV error: ",
    csvParserUnavailable: "The CSV parser is not available.",
    textCsvColumnMismatch:
      "Text mode requires {expected} columns: text + {outputs} outputs, but the CSV contains {actual}.",
    textCsvEmptyText: "Empty text at row",
    textConfigMismatch:
      "The imported alphabet does not match the network input size.",
    presetXorLoaded: "XOR preset loaded (4 examples)",
    presetLinearLoaded: "Linear dataset loaded (200 examples)",
    presetSentimentLoaded:
      "English sentiment loaded (40 sentences: 0 negative, 1 positive)",
    datasetPreviewCount: "First {shown} of {total} rows",
    chartEpochs: "Epochs",
    lossHelp:
      "Loss measures how far predictions are from the desired outputs. Lower values mean the network is learning better.",
    accuracyHelp:
      "Accuracy is the percentage of examples classified correctly. Higher values mean better results.",
    jsonItem: "item",
    jsonItems: "items",
    jsonProperty: "property",
    jsonProperties: "properties",
    jsonExpandNode: "Expand item",
    jsonCollapseNode: "Collapse item",
    trainNoDataset: "Load or choose a preset/CSV before training.",
    standardTraining: "Normal",
    stepTraining: "Step-by-step",
    stepReset: "Reset",
    stepLoadExample: "Load example",
    stepNext: "Next step",
    stepReady: "Ready to start",
    stepExampleLoaded: "New example loaded",
    stepForwardLayer: "Forward pass · layer {layer}",
    stepLossCalculated: "Loss calculated",
    stepBackwardLayer: "Backpropagation · layer {layer} updated",
    stepUpdateComplete: "Update complete · new output calculated",
    stepExampleCounter: "Example",
    stepExample: "Example",
    stepDesired: "Desired output",
    stepCurrent: "Current output",
    stepTargetShort: "target",
    stepNoNetwork: "Configure at least one output layer.",
    inputMismatch: "Input size doesn't match the network.",
    roundedOutput: "Rounded value",
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
        <b>Numeric</b><br>
        • Inputs first, outputs last<br>
        • Optional header row<br>
        • Example: <code>0,1,1</code><br><br>
        <b>Text</b><br>
        • First column = text, followed by outputs<br>
        • Optional header row<br>
        • Example: <code>text,label<br>"useful lesson",1</code>
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
  renderDatasetPreview();
  updateTrainingLanguage();
}

const I18N_HTML = {
  it: {
    palette: "Palette",
    datasetPreview: "Anteprima dataset",
    addHiddenLayer: "Aggiungi layer nascosto",
    importAction: "Importa",
    exportFull: "Rete completa",
    exportArchitecture: "Architettura",
    exportWeights: "Pesi",
    jsonPreview: "Anteprima JSON della rete",
    compact: "Compatto",
    collapse: "Collapse",
    lossHelpAria: "Spiega la loss",
    accuracyHelpAria: "Spiega l'accuracy",
    jsonTreeAria: "Anteprima JSON espandibile",
    dragHere: "Trascina i blocchi qui sotto 👉",
    input: "Input",
    output: "Output",
    hiddenLayer: "Layer nascosto",
    neurons: "Neuroni",
    inputSize: "Dimensione input",
    inputType: "Tipo di input",
    numericInput: "Numerico",
    textInput: "Testo",
    alphabet: "Alfabeto",
    lowercase: "Minuscolo",
    textPredictPlaceholder: "Scrivi il testo da classificare...",
    activation: "Attivazione",
    weightedSum: "Somma pesata",
    inputLabel: "Input",
    valueLabel: "Valore",
    bias: "Bias",
    legendPositive: "Verde = peso positivo",
    legendNegative: "Rosso = peso negativo",
    legendThickness: "Spessore = intensità del peso",
    legendHover: "Passa col mouse sui collegamenti per vedere il peso numerico",
    legendNodeColor: "Colore nodo = livello di attivazione del neurone",
    clear: "Pulisci",
    architecture: "Architettura",
    training: "Training",
    standardTraining: "Normale",
    stepTraining: "Step-by-step",
    stepReset: "Reset",
    stepLoadExample: "Carica esempio",
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
    supervisedData: "Dati per apprendimento supervisionato",
    load: "Carica",
    csvDropText: "Trascina CSV qui",
    noDataset: "Nessun dataset caricato",
    json: "JSON",
    export: "Export",
    download: "Download",
    copy: "Copy",

    linearPreset: "Lineare (x + y > 1)",
    sentimentPreset: "Sentiment inglese (positivo / negativo)",
  },

  en: {
    palette: "Palette",
    datasetPreview: "Dataset preview",
    addHiddenLayer: "Add Hidden Layer",
    importAction: "Import",
    exportFull: "Full network",
    exportArchitecture: "Architecture",
    exportWeights: "Weights",
    jsonPreview: "JSON network preview",
    compact: "Compact",
    collapse: "Collapse",
    lossHelpAria: "Explain loss",
    accuracyHelpAria: "Explain accuracy",
    jsonTreeAria: "Collapsible JSON preview",
    dragHere: "Drag the blocks below 👉",

    input: "Input",
    output: "Output",
    hiddenLayer: "Hidden layer",
    neurons: "Neurons",
    inputSize: "Input size",
    inputType: "Input type",
    numericInput: "Numeric",
    textInput: "Text",
    alphabet: "Alphabet",
    lowercase: "Lowercase",
    textPredictPlaceholder: "Type the text to classify...",
    activation: "Activation",
    weightedSum: "Weighted sum",
    inputLabel: "Input",
    valueLabel: "Value",
    bias: "Bias",
    legendPositive: "Green = positive weight",
    legendNegative: "Red = negative weight",
    legendThickness: "Thickness = weight strength",
    legendHover: "Hover over the connections to see the numeric weight",
    legendNodeColor: "Node color = neuron activation level",
    clear: "Clear",
    architecture: "Architecture",
    training: "Training",
    standardTraining: "Normal",
    stepTraining: "Step-by-step",
    stepReset: "Reset",
    stepLoadExample: "Load example",
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
    supervisedData: "Supervised learning data",
    load: "Load",
    csvDropText: "Drop CSV here",
    noDataset: "No dataset loaded",
    json: "JSON",
    export: "Export",
    download: "Download",
    copy: "Copy",

    linearPreset: "Linear (x + y > 1)",
    sentimentPreset: "English sentiment (positive / negative)",
  },
};
