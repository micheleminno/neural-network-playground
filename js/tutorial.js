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
        selector: "#architectureCard",
        title: "Start with the network recipe",
        body: "This column is the model architecture. Students can see that a neural network is built from layers, neurons, activation functions and bias.",
        action:
          "For a first classroom demo, keep the default 2 inputs, 4 hidden neurons and 1 sigmoid output.",
        placement: "right",
      },
      {
        selector: "#datasetCard",
        title: "Load a tiny learning problem",
        body: "A network needs examples before it can learn. Built-in presets have a known shape: XOR automatically sets 2 inputs and 1 output. With a custom CSV, configure the network first: its input and output counts determine how the columns are interpreted.",
        action: "Load XOR now and notice that the architecture becomes 2 inputs and 1 output.",
        placement: "right",
        primaryLabel: "Load XOR",
        onPrimary: () => {
          const preset = document.getElementById("presetDataset");
          if (preset) preset.value = "xor";
          loadPreset("xor");
        },
      },
      {
        selector: "#trainingCard",
        title: "Train and watch learning happen",
        body: "Training changes weights and bias. The loss chart is the story: if it goes down, the network is learning from the examples.",
        action: "Click Train after XOR is loaded, then watch loss and accuracy.",
        placement: "top",
      },
      {
        selector: "#networkVisCard",
        title: "Read the network visually",
        body: "Green links push activations up, red links push them down. Thicker links are stronger. Neuron color shows activation.",
        action: "Hover over a connection to reveal its numeric weight.",
        placement: "left",
      },
      {
        selector: "#predictionCard",
        title: "Ask the trained model a question",
        body: "Prediction turns the trained network into an experiment. Change x1 and x2 to test what the model believes.",
        action: "Prepare the classic XOR test input: x1 = 1 and x2 = 0.",
        placement: "left",
        primaryLabel: "Set 1, 0",
        onPrimary: () => {
          const inputs = document.querySelectorAll("#testInputs [data-ti]");
          if (inputs[0]) inputs[0].value = "1";
          if (inputs[1]) inputs[1].value = "0";
          predictOnce();
        },
      },
      {
        selector: "#networkFilesCard",
        title: "Save the classroom artifact",
        body: "Saved networks are stored in the cloud under the teacher account, so each teacher sees only their own models.",
        action:
          "When the result is useful, save it with a name students can recognize.",
        placement: "right",
      },
      {
        selector: "#savedNetworksDropdown",
        title: "Reuse and compare",
        body: "The saved networks menu lets a teacher come back later, compare architectures, or prepare examples before class.",
        action:
          "Load old networks from here, then modify and save new versions.",
        placement: "right",
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
        selector: "#architectureCard",
        title: "Parti dalla ricetta della rete",
        body: "Questa colonna descrive l'architettura del modello. Gli studenti vedono che una rete neurale nasce da layer, neuroni, funzioni di attivazione e bias.",
        action:
          "Per la prima dimostrazione in classe, tieni la rete di default: 2 input, 4 neuroni nascosti e 1 output sigmoid.",
        placement: "right",
      },
      {
        selector: "#datasetCard",
        title: "Carica un problema piccolo",
        body: "Una rete ha bisogno di esempi per imparare. I preset hanno una struttura nota: XOR imposta automaticamente 2 input e 1 output. Con un CSV personalizzato, configura prima la rete: il numero di input e output stabilisce come vengono interpretate le colonne.",
        action: "Carica XOR e osserva che l'architettura diventa 2 input e 1 output.",
        placement: "right",
        primaryLabel: "Carica XOR",
        onPrimary: () => {
          const preset = document.getElementById("presetDataset");
          if (preset) preset.value = "xor";
          loadPreset("xor");
        },
      },
      {
        selector: "#trainingCard",
        title: "Allena e osserva l'apprendimento",
        body: "L'allenamento modifica pesi e bias. Il grafico della loss racconta cosa sta succedendo: se scende, la rete sta imparando dagli esempi.",
        action: "Dopo aver caricato XOR, clicca Train e osserva loss e accuracy.",
        placement: "top",
      },
      {
        selector: "#networkVisCard",
        title: "Leggi la rete in modo visuale",
        body: "I collegamenti verdi aumentano l'attivazione, quelli rossi la riducono. Le linee piu' spesse sono piu' forti. Il colore dei neuroni mostra l'attivazione.",
        action: "Passa sopra un collegamento per vedere il peso numerico.",
        placement: "left",
      },
      {
        selector: "#predictionCard",
        title: "Fai una domanda al modello",
        body: "La predizione trasforma la rete allenata in un esperimento. Cambia x1 e x2 per testare cosa ha imparato.",
        action: "Prepara il classico test XOR: x1 = 1 e x2 = 0.",
        placement: "left",
        primaryLabel: "Imposta 1, 0",
        onPrimary: () => {
          const inputs = document.querySelectorAll("#testInputs [data-ti]");
          if (inputs[0]) inputs[0].value = "1";
          if (inputs[1]) inputs[1].value = "0";
          predictOnce();
        },
      },
      {
        selector: "#networkFilesCard",
        title: "Salva il risultato della lezione",
        body: "Le reti salvate finiscono nel cloud sotto l'account del docente: ogni docente vede solo i propri modelli.",
        action:
          "Quando il risultato e' utile, salvalo con un nome riconoscibile per gli studenti.",
        placement: "right",
      },
      {
        selector: "#savedNetworksDropdown",
        title: "Riusa e confronta",
        body: "Il menu delle reti salvate permette di tornare su un esempio, confrontare architetture o preparare modelli prima della lezione.",
        action:
          "Carica da qui una rete salvata, poi modificala e salvala come nuova versione.",
        placement: "right",
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
  return `neurobuilder-tutorial-v2-seen:${userId}`;
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
