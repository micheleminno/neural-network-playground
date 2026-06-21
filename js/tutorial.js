const tutorialCopy = {
  en: {
    step: "Step",
    of: "of",
    back: "Back",
    next: "Next",
    finish: "Finish",
    skip: "Skip",
    close: "Close tutorial",
    actionLabel: "Try this",
    steps: [
      {
        selector: "#networkFilesCard",
        title: "Open, save or exchange a network",
        body: "All network file actions are collected here. You can reopen one of your cloud networks, update it, save a new copy, or import and export JSON files.",
        action: "For now, leave the selector empty and build a new classroom example.",
        placement: "bottom",
      },
      {
        selector: "#architectureCard",
        title: "Start with the network recipe",
        body: "The architecture now reads from left to right: input, hidden layers and output. Inputs can be numeric or text encoded as character frequencies.",
        action:
          "Leave the hidden layer and sigmoid output as they are. The sentiment preset will configure the text input automatically.",
        placement: "bottom",
      },
      {
        selector: "#datasetCard",
        title: "Teach the network with sentences",
        body: "A network learns from labelled examples. This preset contains 40 English sentences: negative sentences are labelled 0 and positive sentences are labelled 1. It also configures the text input automatically.",
        action: "Load English sentiment and notice that the input changes from numbers to text features.",
        placement: "right",
        primaryLabel: "Load sentiment",
        onPrimary: () => {
          const preset = document.getElementById("presetDataset");
          if (preset) preset.value = "sentiment-en";
          loadPreset("sentiment-en");
        },
      },
      {
        selector: "#trainingCard",
        title: "Train and watch learning happen",
        body: "Training changes weights and bias. Normal mode runs automatically; Step-by-step shows one example, each forward layer, the loss and each backward weight update directly on the network.",
        action: "Choose Step-by-step, load an example and use Next step to follow the calculation from left to right and back again.",
        placement: "left",
      },
      {
        selector: "#networkVisCard",
        title: "Read the network visually",
        body: "Green links push activations up, red links push them down. Thicker links are stronger. Neuron color shows activation.",
        action: "Hover over a connection to reveal its numeric weight. During Step-by-step, follow the animated direction across the diagram.",
        placement: "bottom",
      },
      {
        selector: "#predictionCard",
        title: "Ask the trained model a question",
        body: "Predict is now part of the network visualization. The text box sits to the left of the input nodes, so the question and the network response remain visible together.",
        action: 'Try the sentence: "This lesson was clear and useful".',
        placement: "right",
        primaryLabel: "Use example sentence",
        onPrimary: () => {
          const input = document.getElementById("textPredictInput");
          if (input) {
            input.value = "This lesson was clear and useful";
            input.dispatchEvent(new Event("input", { bubbles: true }));
          }
          predictOnce();
        },
      },
      {
        selector: "#jsonCard",
        title: "Inspect the network as data",
        body: "The JSON view exposes architecture, input configuration and weights. It connects the visual model with a format students can inspect, copy and export.",
        action: "Compare a layer in the architecture with its corresponding values in JSON.",
        placement: "top",
      },
      {
        selector: "#btnStartTutorial",
        title: "You can restart this anytime",
        body: "The Tutorial button brings this guide back. Use it when presenting NeuroBuilder to another teacher or a class.",
        action: "Close the tour and start experimenting.",
        placement: "bottom",
      },
    ],
  },
  it: {
    step: "Passo",
    of: "di",
    back: "Indietro",
    next: "Avanti",
    finish: "Fine",
    skip: "Salta",
    close: "Chiudi tutorial",
    actionLabel: "Prova ora",
    steps: [
      {
        selector: "#networkFilesCard",
        title: "Apri, salva o scambia una rete",
        body: "Tutte le operazioni sui file sono raccolte qui. Puoi riaprire una rete dal cloud, aggiornarla, salvarne una nuova copia oppure importare ed esportare file JSON.",
        action: "Per ora lascia vuoto il selettore e costruisci un nuovo esempio didattico.",
        placement: "bottom",
      },
      {
        selector: "#architectureCard",
        title: "Parti dalla ricetta della rete",
        body: "L'architettura ora si legge da sinistra a destra: input, layer nascosti e output. Gli input possono essere numerici oppure testi codificati come frequenze di caratteri.",
        action:
          "Lascia invariati il layer nascosto e l'output sigmoid. Il preset sentiment configurerà automaticamente l'input testuale.",
        placement: "bottom",
      },
      {
        selector: "#datasetCard",
        title: "Insegna alla rete usando delle frasi",
        body: "Una rete impara da esempi etichettati. Questo preset contiene 40 frasi inglesi: le frasi negative hanno etichetta 0 e quelle positive etichetta 1. Configura anche l'input testuale automaticamente.",
        action: "Carica il sentiment inglese e osserva che l'input passa dai numeri alle feature testuali.",
        placement: "right",
        primaryLabel: "Carica sentiment",
        onPrimary: () => {
          const preset = document.getElementById("presetDataset");
          if (preset) preset.value = "sentiment-en";
          loadPreset("sentiment-en");
        },
      },
      {
        selector: "#trainingCard",
        title: "Allena e osserva l'apprendimento",
        body: "L'allenamento modifica pesi e bias. La modalità Normale procede automaticamente; Step-by-step mostra sulla rete un esempio, ogni layer del forward, la loss e ogni aggiornamento dei pesi durante la backpropagation.",
        action: "Scegli Step-by-step, carica un esempio e usa Passo successivo per seguire il calcolo da sinistra a destra e poi al contrario.",
        placement: "left",
      },
      {
        selector: "#networkVisCard",
        title: "Leggi la rete in modo visuale",
        body: "I collegamenti verdi aumentano l'attivazione, quelli rossi la riducono. Le linee piu' spesse sono piu' forti. Il colore dei neuroni mostra l'attivazione.",
        action: "Passa sopra un collegamento per vedere il peso numerico. In Step-by-step segui la direzione animata lungo il diagramma.",
        placement: "bottom",
      },
      {
        selector: "#predictionCard",
        title: "Fai una domanda al modello",
        body: "Predict ora fa parte della visualizzazione. Il box di testo si trova a sinistra dei nodi di input, così domanda e risposta della rete restano visibili insieme.",
        action: 'Prova la frase: "This lesson was clear and useful".',
        placement: "right",
        primaryLabel: "Usa la frase di esempio",
        onPrimary: () => {
          const input = document.getElementById("textPredictInput");
          if (input) {
            input.value = "This lesson was clear and useful";
            input.dispatchEvent(new Event("input", { bubbles: true }));
          }
          predictOnce();
        },
      },
      {
        selector: "#jsonCard",
        title: "Osserva la rete come dati",
        body: "La vista JSON espone architettura, configurazione degli input e pesi. Collega il modello visuale a un formato che gli studenti possono leggere, copiare ed esportare.",
        action: "Confronta un layer dell'architettura con i valori corrispondenti nel JSON.",
        placement: "top",
      },
      {
        selector: "#btnStartTutorial",
        title: "Puoi riaprire la guida quando vuoi",
        body: "Il pulsante Tutorial riapre questa guida. Usalo quando presenti NeuroBuilder ad altri docenti o alla classe.",
        action: "Chiudi il tour e inizia a sperimentare.",
        placement: "bottom",
      },
    ],
  },
};

let tutorialIndex = 0;
let tutorialActive = false;

function currentTutorialCopy() {
  return tutorialCopy[getLang()] || tutorialCopy.en;
}

function tutorialSteps() {
  return currentTutorialCopy().steps;
}

function tutorialStorageKey() {
  const userId = currentSession?.user?.id || "anonymous";
  return `neurobuilder-tutorial-v4-seen:${userId}`;
}

function tutorialEls() {
  return {
    overlay: document.getElementById("tutorialOverlay"),
    highlight: document.getElementById("tutorialHighlight"),
    card: document.getElementById("tutorialCard"),
    progress: document.getElementById("tutorialProgress"),
    counter: document.getElementById("tutorialCounter"),
    title: document.getElementById("tutorialTitle"),
    body: document.getElementById("tutorialBody"),
    action: document.getElementById("tutorialAction"),
    primary: document.getElementById("tutorialPrimary"),
    back: document.getElementById("tutorialBack"),
    next: document.getElementById("tutorialNext"),
    skip: document.getElementById("tutorialSkip"),
    close: document.getElementById("tutorialClose"),
  };
}

function ensureTutorialDom() {
  if (document.getElementById("tutorialOverlay")) return;

  const overlay = document.createElement("div");
  overlay.id = "tutorialOverlay";
  overlay.className = "tutorial-overlay d-none";
  overlay.innerHTML = `
    <div id="tutorialHighlight" class="tutorial-highlight"></div>
    <section id="tutorialCard" class="tutorial-card" aria-live="polite">
      <div class="tutorial-progress-track mb-3">
        <div id="tutorialProgress" class="tutorial-progress"></div>
      </div>
      <div class="d-flex justify-content-between align-items-start gap-3 mb-2">
        <div id="tutorialCounter" class="tutorial-counter"></div>
        <button id="tutorialClose" class="btn btn-sm btn-outline-light" type="button">
          <i class="bi bi-x-lg"></i>
        </button>
      </div>
      <h5 id="tutorialTitle" class="mb-2"></h5>
      <p id="tutorialBody" class="mb-2"></p>
      <div id="tutorialAction" class="tutorial-action mb-3"></div>
      <button id="tutorialPrimary" class="btn btn-info w-100 mb-3 d-none" type="button"></button>
      <div class="d-flex justify-content-between gap-2">
        <button id="tutorialSkip" class="btn btn-link text-secondary px-0" type="button"></button>
        <div class="d-flex gap-2">
          <button id="tutorialBack" class="btn btn-outline-light" type="button"></button>
          <button id="tutorialNext" class="btn btn-primary" type="button"></button>
        </div>
      </div>
    </section>
  `;

  document.body.appendChild(overlay);

  tutorialEls().close?.addEventListener("click", () => endTutorial(true));
  tutorialEls().skip?.addEventListener("click", () => endTutorial(true));
  tutorialEls().back?.addEventListener("click", () => {
    showTutorialStep(tutorialIndex - 1);
  });
  tutorialEls().next?.addEventListener("click", () => {
    const steps = tutorialSteps();
    if (tutorialIndex >= steps.length - 1) {
      endTutorial(true);
    } else {
      showTutorialStep(tutorialIndex + 1);
    }
  });
  tutorialEls().primary?.addEventListener("click", () => {
    const step = tutorialSteps()[tutorialIndex];
    step?.onPrimary?.();
    showTutorialStep(tutorialIndex);
  });

  document.addEventListener("keydown", (event) => {
    if (!tutorialActive) return;

    if (event.key === "Escape") {
      endTutorial(true);
    } else if (event.key === "ArrowRight") {
      event.preventDefault();
      showTutorialStep(tutorialIndex + 1);
    } else if (event.key === "ArrowLeft") {
      event.preventDefault();
      showTutorialStep(tutorialIndex - 1);
    }
  });
}

function positionTutorialCard(target, placement) {
  const { card } = tutorialEls();
  if (!card) return;

  const rect = target.getBoundingClientRect();
  const gap = 16;
  const cardRect = card.getBoundingClientRect();
  let left = rect.right + gap;
  let top = rect.top;

  if (placement === "left") {
    left = rect.left - cardRect.width - gap;
  } else if (placement === "top") {
    left = rect.left;
    top = rect.top - cardRect.height - gap;
  } else if (placement === "bottom") {
    left = rect.left;
    top = rect.bottom + gap;
  }

  left = Math.max(12, Math.min(left, window.innerWidth - cardRect.width - 12));
  top = Math.max(12, Math.min(top, window.innerHeight - cardRect.height - 12));

  card.style.left = `${left}px`;
  card.style.top = `${top}px`;
}

function positionTutorialHighlight() {
  if (!tutorialActive) return;

  const step = tutorialSteps()[tutorialIndex];
  const target = step && document.querySelector(step.selector);
  const { highlight } = tutorialEls();

  if (!target || !highlight) return;

  const rect = target.getBoundingClientRect();
  const pad = 8;

  highlight.style.left = `${Math.max(6, rect.left - pad)}px`;
  highlight.style.top = `${Math.max(6, rect.top - pad)}px`;
  highlight.style.width = `${Math.min(
    window.innerWidth - 12,
    rect.width + pad * 2,
  )}px`;
  highlight.style.height = `${Math.min(
    window.innerHeight - 12,
    rect.height + pad * 2,
  )}px`;

  positionTutorialCard(target, step.placement);
}

function showTutorialStep(index) {
  if (!tutorialActive) return;

  const copy = currentTutorialCopy();
  const steps = tutorialSteps();
  tutorialIndex = Math.max(0, Math.min(index, steps.length - 1));
  const step = steps[tutorialIndex];
  const target = document.querySelector(step.selector);
  const {
    overlay,
    highlight,
    progress,
    counter,
    title,
    body,
    action,
    primary,
    back,
    next,
    skip,
    close,
  } = tutorialEls();

  if (!overlay || !highlight || !target) return;

  target.scrollIntoView({ behavior: "auto", block: "center", inline: "center" });

  requestAnimationFrame(() => {
    const progressPct = ((tutorialIndex + 1) / steps.length) * 100;

    if (progress) progress.style.width = `${progressPct}%`;
    if (counter) {
      counter.textContent = `${copy.step} ${tutorialIndex + 1} ${copy.of} ${
        steps.length
      }`;
    }
    if (title) title.textContent = step.title;
    if (body) body.textContent = step.body;
    if (action) {
      action.innerHTML = `<strong>${copy.actionLabel}:</strong> ${step.action}`;
    }
    if (primary) {
      primary.classList.toggle("d-none", !step.primaryLabel);
      primary.textContent = step.primaryLabel || "";
    }
    if (back) {
      back.textContent = copy.back;
      back.disabled = tutorialIndex === 0;
    }
    if (next) {
      next.textContent = tutorialIndex === steps.length - 1 ? copy.finish : copy.next;
    }
    if (skip) skip.textContent = copy.skip;
    if (close) close.setAttribute("aria-label", copy.close);

    positionTutorialHighlight();
  });
}

function startTutorial(force = false) {
  if (tutorialActive) return;
  if (!force && localStorage.getItem(tutorialStorageKey()) === "1") return;

  ensureTutorialDom();
  tutorialActive = true;
  tutorialIndex = 0;

  tutorialEls().overlay?.classList.remove("d-none");
  showTutorialStep(0);
}

function endTutorial(markSeen = false) {
  tutorialActive = false;

  if (markSeen) {
    localStorage.setItem(tutorialStorageKey(), "1");
  }

  tutorialEls().overlay?.classList.add("d-none");
}

function refreshActiveTutorialLanguage() {
  if (tutorialActive) showTutorialStep(tutorialIndex);
}

function initTutorialControls() {
  document.getElementById("btnStartTutorial")?.addEventListener("click", () => {
    startTutorial(true);
  });

  window.addEventListener("resize", () => {
    positionTutorialHighlight();
  });

  window.addEventListener(
    "scroll",
    () => {
      positionTutorialHighlight();
    },
    true,
  );
}

function resetTutorialForCurrentUser() {
  localStorage.removeItem(tutorialStorageKey());
}

function startTutorialIfNeeded() {
  setTimeout(() => startTutorial(false), 650);
}
