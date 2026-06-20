// ========= NeuroBuilder - script unico IT/EN =========
// Richiede nell'HTML:
// <body data-lang="it"> oppure <body data-lang="en">
// bottone lingua: #btnLangToggle
// tutti gli id già presenti nel tuo HTML unico.

// ========= Utility =========
const $ = (s) => document.querySelector(s);
const $$ = (s) => Array.from(document.querySelectorAll(s));

function getLang() {
  return document.body?.dataset?.lang === "en" ? "en" : "it";
}

function setLang(lang) {
  const safeLang = lang === "en" ? "en" : "it";

  document.body.dataset.lang = safeLang;
  document.documentElement.lang = safeLang;

  localStorage.setItem("neurobuilder-lang", safeLang);

  applyI18n();

  // forza rebuild UI
  renderArchitecture();
  renderTestInputs();
  renderNNVis();
  updateJSON();
  updateNetworkTitle();

  if (typeof refreshActiveTutorialLanguage === "function") {
    refreshActiveTutorialLanguage();
  }
}

function t(key) {
  const lang = getLang();
  return I18N[lang]?.[key] ?? I18N.it[key] ?? key;
}

function rng(seed = 123) {
  let s = seed >>> 0;
  return function () {
    s ^= s << 13;
    s ^= s >>> 17;
    s ^= s << 5;
    s >>>= 0;
    return (s % 1_000_000) / 1_000_000;
  };
}

function shuffleInPlace(arr, rand) {
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
}

function clamp(x, min, max) {
  return Math.max(min, Math.min(max, x));
}
