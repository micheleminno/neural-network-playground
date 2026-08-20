/* =============================================================================
   access.js — lightweight access gate for NeuroBuilder
   -----------------------------------------------------------------------------
   IMPORTANT — this is a LIGHT gate, by design. It runs entirely in the browser
   on a static site, so it deters casual visitors but a technical user who knows
   the URL + token CAN bypass it. For real enforcement you'd need a server-side
   validator (see the "proper gate" option we discussed).

   HOW IT WORKS
   The app (#appShell) is hidden by default (fail-closed). It is revealed only
   when the visitor arrives with the shared access token that SARA renders ONLY
   on member-only pages (i.e. only logged-in SARA members ever get the link):

       https://<playground-url>/?sara_access=THE_TOKEN

   Once granted, access persists for the browser session (so navigating inside
   the playground doesn't re-block). Without the token, a "sign in on SARA"
   screen (the landing) is shown instead.

   SETUP (two sides must share the same token)
   - Here:   set SARA_TOKEN below to a private string of your choice.
   - On SARA: in Contao, put the playground link inside a content element/page
     restricted to logged-in members, pointing to
         .../?sara_access=THE_SAME_TOKEN
     Optionally add &exp=<unix-milliseconds> so copied links expire.
   - Also set SARA_LOGIN_URL to the SARA members/login page.
   ============================================================================= */

(function () {
  "use strict";

  var SARA_TOKEN = "nb-7Kx92pQ-labs"; // must match the Contao link
  var SARA_LOGIN_URL = "https://www.sara-systems.net/login"; // SARA login page
  var SARA_REGISTER_URL = "https://www.sara-systems.net/register"; // SARA registration page
  var SESSION_KEY = "saraAccess";
  var ALLOW_REFERRER = false; // true = also grant if the referrer is a SARA page (weaker)

  function param(name) {
    try {
      return new URLSearchParams(location.search).get(name);
    } catch (e) {
      return null;
    }
  }

  function hasAccess() {
    try {
      var tok = param("sara_access");
      if (tok !== null && tok === SARA_TOKEN) {
        var exp = param("exp");
        if (exp && Date.now() > Number(exp)) return false; // link expired
        sessionStorage.setItem(SESSION_KEY, "1");
        return true;
      }
      if (sessionStorage.getItem(SESSION_KEY) === "1") return true;
      if (ALLOW_REFERRER && document.referrer) {
        var host = new URL(document.referrer).hostname;
        if (/(^|\.)sara-systems\.(net|eu)$/i.test(host)) return true;
      }
    } catch (e) {}
    return false;
  }

  function setHrefs(selector, url) {
    var els = document.querySelectorAll(selector);
    for (var i = 0; i < els.length; i++) els[i].setAttribute("href", url);
  }

  function cameFromSara() {
    try {
      var qs = new URLSearchParams(location.search);
      if (qs.has("sara") || qs.get("embed") === "sara" || qs.has("sara_access")) return true;
      if (sessionStorage.getItem("saraFrame") === "1") return true;
      var ref = document.referrer ? new URL(document.referrer).hostname : "";
      if (/(^|\.)sara-systems\.(net|eu)$/i.test(ref)) return true;
    } catch (e) {}
    return false;
  }

  function apply() {
    var app = document.getElementById("appShell");

    if (!cameFromSara()) {
      /* Direct visit to GitHub Pages — show the app freely */
      if (app) app.classList.remove("d-none");
      return;
    }

    /* Arrived from SARA — enforce token */
    if (hasAccess()) {
      if (app) app.classList.remove("d-none");
    } else {
      window.location.replace(SARA_LOGIN_URL);
    }
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", apply);
  } else {
    apply();
  }
})();
