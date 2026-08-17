/* =============================================================================
   sara-frame.js
   Shared SARA header + footer for the Interactive Learning Labs modules.
   Styled to match sara-systems.net (white header, navy #003963, Poppins).

   HOW TO USE
   ----------
   Add ONE line just before the closing </body> tag of your index.html:

       <script src="sara-frame.js" defer></script>

   Nothing else in your page needs to change. The script prepends a SARA
   header and appends a SARA footer. All its CSS is prefixed with
   "sara-frame-" so it will not interfere with the playground's own styles.

   REUSE ON OTHER MODULES
   ----------------------
   Copy this same file into each future module (transformers, reinforcement
   learning, ...) and change only the MODULE object below. Later, if you want
   a single shared source, host this file on sara-systems.net and let every
   module load it from there.
   ============================================================================= */

(function () {
  "use strict";

  /* ---- Per-module configuration: change these per module ---- */
  var MODULE = {
    name: "Neural Network Playground",
    tagline: "Interactive Learning Labs",
    author: "Michele Minno"
  };

  /* ---- When to show the SARA frame ----------------------------------------
     The header/footer appear ONLY when the visitor arrives from SARA.
     A direct visit to the bare GitHub Pages URL stays clean (no frame).

     It shows the frame when ANY of these is true:
       - the URL has ?sara=1 (or ?embed=sara)  <-- link SARA uses to point here
       - the referring page is on sara-systems.net / .eu
       - the frame was already activated earlier in this browsing session
     Set FORCE_SHOW to true only if you ever want it always on. */
  var FORCE_SHOW = false;

  function cameFromSara() {
    if (FORCE_SHOW) return true;
    try {
      var qs = new URLSearchParams(location.search);
      if (qs.has("sara") || qs.get("embed") === "sara" || qs.has("sara_access")) {
        sessionStorage.setItem("saraFrame", "1");
        return true;
      }
      if (sessionStorage.getItem("saraFrame") === "1") return true;
      var ref = document.referrer ? new URL(document.referrer).hostname : "";
      if (/(^|\.)sara-systems\.(net|eu)$/i.test(ref)) {
        sessionStorage.setItem("saraFrame", "1");
        return true;
      }
    } catch (e) {}
    return false;
  }

  /* ---- SARA brand (verified live from sara-systems.net) ---- */
  var SARA = {
    home: "https://www.sara-systems.net/",
    logo: "https://www.sara-systems.net/files/layout/logos/sara-logo.svg",
    favicon: "https://www.sara-systems.net/favicon.ico",
    email: "cooperation@sara-systems.eu",
    nav: [
      { label: "Profile", href: "https://www.sara-systems.net/profile" },
      { label: "AI & Simulation", href: "https://www.sara-systems.net/artificial-intelligence" },
      { label: "Consulting & Education", href: "https://www.sara-systems.net/education" },
      { label: "Open Source Software", href: "https://www.sara-systems.net/open" },
      { label: "Job Offers", href: "https://www.sara-systems.net/job-offers" }
    ],
    legal:   "https://www.sara-systems.net/legal-informations",
    privacy: "https://www.sara-systems.net/data-privacy",
    terms:   "https://www.sara-systems.net/terms-conditions",
    contact: "https://www.sara-systems.net/contact"
  };

  /* ---- SARA brand colours ---- */
  var CSS =
    ':root{' +
      '--sara-navy:#003963;' +
      '--sara-navy-2:#00263f;' +   /* darker navy for the footer base */
      '--sara-line:#e3e8ee;' +
      '--sara-muted:#b9c6d4;' +
    '}' +

    '.sara-frame-header,.sara-frame-footer{' +
      'font-family:Poppins,Helvetica,Arial,sans-serif;' +
      'box-sizing:border-box;line-height:1.5;}' +
    '.sara-frame-header *,.sara-frame-footer *{box-sizing:border-box;}' +

    /* Header — white, two rows (logo, then nav) matching sara-systems.net */
    '.sara-frame-header{' +
      'background:#ffffff;color:var(--sara-navy);' +
      'padding:18px 28px 12px;border-bottom:2px solid var(--sara-navy);}' +
    '.sara-frame-brand{display:inline-flex;align-items:center;gap:16px;text-decoration:none;color:inherit;}' +
    '.sara-frame-brand img{height:52px;width:auto;display:block;}' +
    '.sara-frame-brand-txt{display:flex;flex-direction:column;line-height:1.15;' +
      'border-left:2px solid var(--sara-line);padding-left:16px;}' +
    '.sara-frame-brand-txt .m{font-weight:600;font-size:15px;color:var(--sara-navy);}' +
    '.sara-frame-brand-txt .t{font-size:10.5px;letter-spacing:.9px;text-transform:uppercase;color:#6f8296;}' +
    '.sara-frame-nav{display:flex;align-items:center;flex-wrap:wrap;justify-content:flex-end;' +
      'gap:10px 26px;margin-top:12px;}' +
    '.sara-frame-nav a{color:var(--sara-navy);text-decoration:none;font-size:14px;font-weight:500;}' +
    '.sara-frame-nav a:hover{text-decoration:underline;}' +
    '.sara-frame-back{border:1.5px solid var(--sara-navy);border-radius:4px;padding:6px 13px !important;' +
      'font-weight:600;}' +
    '.sara-frame-back:hover{background:var(--sara-navy);color:#fff !important;text-decoration:none !important;}' +

    /* Footer — minimal, matching sara-systems.net (light, slim link row) */
    '.sara-frame-footer{' +
      'background:#ffffff;color:#6f8296;' +
      'padding:18px 24px 22px;margin-top:44px;' +
      'border-top:2px solid var(--sara-navy);' +
      'display:flex;flex-wrap:wrap;gap:8px 22px;justify-content:center;align-items:center;text-align:center;}' +
    '.sara-frame-flinks{display:flex;flex-wrap:wrap;gap:6px 20px;justify-content:center;width:100%;}' +
    '.sara-frame-flinks a{color:var(--sara-navy);text-decoration:none;font-size:13px;font-weight:500;}' +
    '.sara-frame-flinks a:hover{text-decoration:underline;}' +
    '.sara-frame-credit{width:100%;font-size:12px;color:#8b9aa8;}' +
    '.sara-frame-credit a{color:var(--sara-navy);text-decoration:none;}' +
    '.sara-frame-credit a:hover{text-decoration:underline;}' +

    '@media (max-width:640px){' +
      '.sara-frame-header{text-align:center;}' +
      '.sara-frame-brand{justify-content:center;}' +
      '.sara-frame-nav{justify-content:center;}' +
    '}';

  function loadPoppins() {
    if (document.querySelector('link[data-sara-font]')) return;
    var l = document.createElement("link");
    l.rel = "stylesheet";
    l.setAttribute("data-sara-font", "1");
    l.href = "https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600&display=swap";
    document.head.appendChild(l);
  }

  function setSaraFavicon() {
    if (!SARA.favicon) return;
    var links = document.querySelectorAll('link[rel~="icon"]');
    for (var i = 0; i < links.length; i++) links[i].parentNode.removeChild(links[i]);
    var l = document.createElement("link");
    l.rel = "icon";
    l.href = SARA.favicon;
    document.head.appendChild(l);
  }

  function el(html) {
    var d = document.createElement("div");
    d.innerHTML = html.trim();
    return d.firstChild;
  }

  function navHtml() {
    return SARA.nav.map(function (n) {
      return '<a href="' + n.href + '">' + n.label + '</a>';
    }).join("");
  }

  function build() {
    loadPoppins();
    setSaraFavicon();

    var style = document.createElement("style");
    style.textContent = CSS;
    document.head.appendChild(style);

    var header = el(
      '<header class="sara-frame-header">' +
        '<a class="sara-frame-brand" href="' + SARA.home + '">' +
          '<img src="' + SARA.logo + '" alt="SARA">' +
          '<span class="sara-frame-brand-txt">' +
            '<span class="m">' + MODULE.name + '</span>' +
            '<span class="t">' + MODULE.tagline + '</span>' +
          '</span>' +
        '</a>' +
        '<nav class="sara-frame-nav">' +
          navHtml() +
          '<a class="sara-frame-back" href="' + SARA.home + '">&larr; Back to SARA</a>' +
        '</nav>' +
      '</header>'
    );
    document.body.insertBefore(header, document.body.firstChild);

    var footer = el(
      '<footer class="sara-frame-footer">' +
        '<div class="sara-frame-flinks">' +
          '<a href="' + SARA.legal + '">Legal Informations</a>' +
          '<a href="' + SARA.privacy + '">Data Privacy</a>' +
          '<a href="' + SARA.terms + '">Terms &amp; Conditions</a>' +
          '<a href="' + SARA.contact + '">Contact</a>' +
        '</div>' +
        '<div class="sara-frame-credit">' +
          MODULE.name + ' \u2014 an Interactive Learning Labs module by ' + MODULE.author + '. ' +
          '\u00A9 ' + new Date().getFullYear() + ' SARA \u2014 The Science Company.' +
        '</div>' +
      '</footer>'
    );
    document.body.appendChild(footer);
  }

  function init() {
    if (!cameFromSara()) return;   /* direct GitHub Pages visit: stay clean */
    build();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
