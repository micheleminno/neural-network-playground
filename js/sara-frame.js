/* =============================================================================
   sara-frame.js
   Adds the real SARA header + footer when the user arrives from sara-systems.net.
   CSS mirrors the live sara-systems.net layout (verified via curl).

   HOW TO USE
   ----------
   The script is loaded in <head> of index.html (no defer needed — it registers
   a DOMContentLoaded listener and is safe to load early).

   REUSE ON OTHER MODULES
   ----------------------
   Copy this file into each future module and change only the MODULE object below.
   ============================================================================= */

(function () {
  "use strict";

  /* ---- Per-module configuration ---- */
  var MODULE = {
    name: "Neural Network Playground",
    backLabel: "← Learning Labs",
  };

  /* ---- When to show the SARA frame ----------------------------------------
     Shows ONLY when the visitor arrives from SARA:
       - URL has ?sara or ?sara_access
       - referrer is sara-systems.net / .eu
       - sessionStorage flag set by a previous page in this session
     Direct GitHub Pages visits stay clean (no frame). */
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

  /* ---- SARA URLs (verified live) ---- */
  var SARA = {
    home:    "https://www.sara-systems.net/learning-labs",
    logo:    "https://www.sara-systems.net/files/layout/logos/sara-logo.svg",
    favicon: "https://www.sara-systems.net/favicon.ico",
    nav: [
      { label: "Profile",                href: "https://www.sara-systems.net/profile" },
      { label: "AI & Simulation",        href: "https://www.sara-systems.net/artificial-intelligence" },
      { label: "Consulting & Education", href: "https://www.sara-systems.net/education" },
      { label: "Open Source Software",   href: "https://www.sara-systems.net/open" },
      { label: "Job Offers",             href: "https://www.sara-systems.net/job-offers" },
    ],
    legal:   "https://www.sara-systems.net/legal-informations",
    privacy: "https://www.sara-systems.net/data-privacy",
    terms:   "https://www.sara-systems.net/terms-conditions",
    contact: "https://www.sara-systems.net/contact",
  };

  /* ---- CSS — mirrors the real sara-systems.net layout ---- */
  var CSS = [
    /* Reset for injected elements */
    "#sara-header,#sara-footer{box-sizing:border-box;font-family:Poppins,Helvetica,Arial,sans-serif;color:#003963;}",
    "#sara-header *,#sara-footer *{box-sizing:border-box;}",

    /* ---- Header ---- */
    /* Outer wrapper: white, max-width 1000px, centered */
    "#sara-header{background:#ffffff;border-bottom:1px solid #d5dde6;}",
    "#sara-header .inside{max-width:1000px;min-height:200px;margin:0 auto;padding:0 20px;position:relative;}",

    /* Logo: float left, 300px wide, padded top+right as on real site */
    "#sara-header #logo{float:left;width:300px;padding-top:60px;padding-right:60px;}",
    "#sara-header #logo a{display:block;}",
    "#sara-header #logo img{display:block;width:100%;height:auto;}",

    /* Nav: sits to the right of the logo */
    "#sara-header .mainnavi{overflow:hidden;}",  /* BFC next to the float */
    "#sara-header .mainnavi ul{list-style:none;margin:0;padding:0;border-bottom:14px solid #a1ccdb;display:flex;flex-wrap:wrap;align-items:center;gap:0 4px;}",
    "#sara-header .mainnavi ul li a,#sara-header .mainnavi ul li span{display:block;padding:12px 14px;text-decoration:none;color:#273476;font-size:14px;font-weight:500;}",
    "#sara-header .mainnavi ul li a:hover{text-decoration:underline;}",

    /* Back button styled like a nav pill */
    "#sara-header .sara-back{border:1.5px solid #273476;border-radius:4px;margin-left:auto;white-space:nowrap;}",
    "#sara-header .sara-back a{font-weight:600 !important;}",
    "#sara-header .sara-back a:hover{background:#273476;color:#fff !important;text-decoration:none !important;}",

    /* Clearfix */
    "#sara-header .inside::after{content:'';display:table;clear:both;}",

    /* ---- Footer ---- */
    "#sara-footer{background:#ffffff;border-top:1px solid #d5dde6;margin-top:44px;}",
    "#sara-footer .inside{max-width:1000px;margin:0 auto;padding:16px 20px;display:flex;flex-wrap:wrap;gap:6px 20px;justify-content:center;}",
    "#sara-footer .mod_customnav ul{list-style:none;margin:0;padding:0;display:flex;flex-wrap:wrap;gap:6px 20px;justify-content:center;}",
    "#sara-footer .mod_customnav ul li a{color:#273476;text-decoration:none;font-size:13px;font-weight:500;}",
    "#sara-footer .mod_customnav ul li a:hover{text-decoration:underline;}",

    /* Responsive */
    "@media(max-width:700px){",
    "#sara-header #logo{float:none;width:auto;padding:20px 0 10px;}",
    "#sara-header .mainnavi ul{justify-content:center;}",
    "}",
  ].join("");

  /* ---- Helpers ---- */

  function setSaraFavicon() {
    var links = document.querySelectorAll('link[rel~="icon"]');
    for (var i = 0; i < links.length; i++) links[i].parentNode.removeChild(links[i]);
    var l = document.createElement("link");
    l.rel = "icon";
    l.href = SARA.favicon;
    document.head.appendChild(l);
  }

  function loadPoppins() {
    if (document.querySelector("link[data-sara-font]")) return;
    var l = document.createElement("link");
    l.rel = "stylesheet";
    l.setAttribute("data-sara-font", "1");
    l.href = "https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600&display=swap";
    document.head.appendChild(l);
  }

  function navItems() {
    return SARA.nav.map(function (n) {
      return '<li><a href="' + n.href + '">' + n.label + "</a></li>";
    }).join("") +
    '<li class="sara-back"><a href="' + SARA.home + '">' + MODULE.backLabel + "</a></li>";
  }

  function build() {
    loadPoppins();
    setSaraFavicon();

    var style = document.createElement("style");
    style.textContent = CSS;
    document.head.appendChild(style);

    /* Header — mirrors real SARA structure */
    var header = document.createElement("header");
    header.id = "sara-header";
    header.innerHTML =
      '<div class="inside">' +
        '<div id="logo"><a href="' + SARA.home + '"><img src="' + SARA.logo + '" alt="SARA Systems"></a></div>' +
        '<nav class="mod_navigation mainnavi block">' +
          '<ul class="level_1">' + navItems() + "</ul>" +
        "</nav>" +
      "</div>";
    document.body.insertBefore(header, document.body.firstChild);

    /* Footer — mirrors real SARA structure */
    var footer = document.createElement("footer");
    footer.id = "sara-footer";
    footer.innerHTML =
      '<div class="inside">' +
        '<nav class="mod_customnav block">' +
          '<ul class="level_1">' +
            '<li><a href="' + SARA.legal   + '">Legal Informations</a></li>' +
            '<li><a href="' + SARA.privacy + '">Data Privacy</a></li>' +
            '<li><a href="' + SARA.terms   + '">Terms &amp; Conditions</a></li>' +
            '<li><a href="' + SARA.contact + '">Contact</a></li>' +
          "</ul>" +
        "</nav>" +
      "</div>";
    document.body.appendChild(footer);
  }

  function init() {
    if (!cameFromSara()) return;
    build();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
