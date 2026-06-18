const tutorialSteps = [
  {
    selector: "#architecture",
    title: "1. Start from the architecture",
    body: "Here you design the neural network: input layer, hidden layers, output layer, neurons, activation functions and bias.",
    action: "Look at the default 2-4-1 network before changing anything.",
    placement: "right",
  },
  {
    selector: "#btnAddHidden",
    title: "2. Add capacity",
    body: "Hidden layers let the network learn patterns that are not just straight lines.",
    action: "Click Add Hidden Layer when you want to make the model more expressive.",
    placement: "right",
    advanceOnClick: true,
  },
  {
    selector: "#presetDataset",
    title: "3. Choose data",
    body: "Training needs examples. Start with XOR or Linear Separation before importing your own CSV.",
    action: "Open this menu and choose a preset dataset.",
    placement: "right",
  },
  {
    selector: "#btnTrain",
    title: "4. Train the network",
    body: "Training adjusts weights and bias to reduce loss. The chart shows whether learning is improving.",
    action: "After selecting a dataset, click Train.",
    placement: "top",
  },
  {
    selector: "#nnVis",
    title: "5. Read the network",
    body: "Green connections are positive weights, red connections are negative. Thicker lines mean stronger weights.",
    action: "Hover a connection to inspect its numeric weight.",
    placement: "left",
  },
  {
    selector: "#btnPredict",
    title: "6. Test a prediction",
    body: "After training, enter custom input values and ask the network for an output.",
    action: "Change the x values, then click Predici / Predict.",
    placement: "left",
  },
  {
    selector: "#btnSaveNetwork",
    title: "7. Save your work",
    body: "Saved networks are stored in the cloud and associated with your account.",
    action: "Click Save As New when you want this network to appear in your saved list.",
    placement: "right",
  },
  {
    selector: "#savedNetworksDropdown",
    title: "8. Come back later",
    body: "This menu shows only your networks. Load one to continue teaching or experimenting from where you left off.",
    action: "Use this dropdown to reopen a saved network.",
    placement: "right",
  },
];

let tutorialIndex = 0;
let tutorialActive = false;

function tutorialStorageKey() {
  const userId = currentSession?.user?.id || "anonymous";
  return `neurobuilder-tutorial-seen:${userId}`;
}

function tutorialEls() {
  return {
    overlay: document.getElementById("tutorialOverlay"),
    highlight: document.getElementById("tutorialHighlight"),
    card: document.getElementById("tutorialCard"),
    counter: document.getElementById("tutorialCounter"),
    title: document.getElementById("tutorialTitle"),
    body: document.getElementById("tutorialBody"),
    action: document.getElementById("tutorialAction"),
    back: document.getElementById("tutorialBack"),
    next: document.getElementById("tutorialNext"),
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
      <div class="d-flex justify-content-between align-items-start gap-3 mb-2">
        <div id="tutorialCounter" class="tutorial-counter"></div>
        <button id="tutorialClose" class="btn btn-sm btn-outline-light" type="button" aria-label="Close tutorial">
          <i class="bi bi-x-lg"></i>
        </button>
      </div>
      <h5 id="tutorialTitle" class="mb-2"></h5>
      <p id="tutorialBody" class="mb-2"></p>
      <div id="tutorialAction" class="tutorial-action mb-3"></div>
      <div class="d-flex justify-content-between gap-2">
        <button id="tutorialBack" class="btn btn-outline-light" type="button">Back</button>
        <button id="tutorialNext" class="btn btn-primary" type="button">Next</button>
      </div>
    </section>
  `;

  document.body.appendChild(overlay);

  document.getElementById("tutorialClose")?.addEventListener("click", () => {
    endTutorial(true);
  });
  document.getElementById("tutorialBack")?.addEventListener("click", () => {
    showTutorialStep(tutorialIndex - 1);
  });
  document.getElementById("tutorialNext")?.addEventListener("click", () => {
    if (tutorialIndex >= tutorialSteps.length - 1) {
      endTutorial(true);
    } else {
      showTutorialStep(tutorialIndex + 1);
    }
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

  document.addEventListener(
    "click",
    (event) => {
      if (!tutorialActive) return;

      const step = tutorialSteps[tutorialIndex];
      if (!step?.advanceOnClick) return;

      const target = document.querySelector(step.selector);
      if (target && target.contains(event.target)) {
        setTimeout(() => showTutorialStep(tutorialIndex + 1), 150);
      }
    },
    true,
  );
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

function showTutorialStep(index) {
  if (!tutorialActive) return;

  tutorialIndex = Math.max(0, Math.min(index, tutorialSteps.length - 1));
  const step = tutorialSteps[tutorialIndex];
  const target = document.querySelector(step.selector);
  const { overlay, highlight, counter, title, body, action, back, next } =
    tutorialEls();

  if (!overlay || !highlight || !target) return;

  target.scrollIntoView({ behavior: "smooth", block: "center", inline: "center" });

  requestAnimationFrame(() => {
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

    if (counter) {
      counter.textContent = `Step ${tutorialIndex + 1} of ${tutorialSteps.length}`;
    }
    if (title) title.textContent = step.title;
    if (body) body.textContent = step.body;
    if (action) action.textContent = step.action;
    if (back) back.disabled = tutorialIndex === 0;
    if (next) {
      next.textContent =
        tutorialIndex === tutorialSteps.length - 1 ? "Finish" : "Next";
    }

    positionTutorialCard(target, step.placement);
  });
}

function startTutorial(force = false) {
  if (tutorialActive) return;
  if (!force && localStorage.getItem(tutorialStorageKey()) === "1") return;

  ensureTutorialDom();
  tutorialActive = true;
  tutorialIndex = 0;

  const { overlay } = tutorialEls();
  overlay?.classList.remove("d-none");
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
    if (tutorialActive) showTutorialStep(tutorialIndex);
  });
}

function startTutorialIfNeeded() {
  setTimeout(() => startTutorial(false), 450);
}
