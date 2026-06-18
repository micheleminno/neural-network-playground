function setAuthMessage(message, mode = "info") {
  const el = document.getElementById("authMessage");
  if (!el) return;

  el.textContent = message || "";
  el.className = `auth-message mt-3 auth-message-${mode}`;
}

function setAuthBusy(isBusy) {
  [
    "btnGoogleLogin",
    "signupFirstName",
    "signupLastName",
    "signupSubject",
    "signupEmail",
    "signupPhone",
    "signupPassword",
    "profileFirstName",
    "profileLastName",
    "profileSubject",
    "profilePhone",
    "loginEmail",
    "loginPassword",
  ].forEach((id) => {
    const el = document.getElementById(id);
    if (el) el.disabled = isBusy;
  });

  document
    .querySelectorAll("#landingPage button[type='submit']")
    .forEach((button) => {
      button.disabled = isBusy;
    });
}

function showLanding() {
  document.getElementById("landingPage")?.classList.remove("d-none");
  document.getElementById("appShell")?.classList.add("d-none");
  document.getElementById("appAcknowledgment")?.classList.add("d-none");
}

function hasCompleteProfile(profile) {
  return Boolean(
    profile?.first_name && profile?.last_name && profile?.teaching_subject,
  );
}

function showAuthForms() {
  document.getElementById("authTabs")?.classList.remove("d-none");
  document.querySelector(".auth-divider")?.classList.remove("d-none");
  document.getElementById("btnGoogleLogin")?.classList.remove("d-none");
  document.querySelector(".tab-content")?.classList.remove("d-none");
  document.getElementById("completeProfileForm")?.classList.add("d-none");
}

function activateAuthTab(tabName) {
  const signupTab = document.getElementById("signup-tab");
  const loginTab = document.getElementById("login-tab");
  const signupPane = document.getElementById("signupPane");
  const loginPane = document.getElementById("loginPane");
  const showLogin = tabName === "login";

  signupTab?.classList.toggle("active", !showLogin);
  signupTab?.setAttribute("aria-selected", String(!showLogin));
  loginTab?.classList.toggle("active", showLogin);
  loginTab?.setAttribute("aria-selected", String(showLogin));

  signupPane?.classList.toggle("show", !showLogin);
  signupPane?.classList.toggle("active", !showLogin);
  loginPane?.classList.toggle("show", showLogin);
  loginPane?.classList.toggle("active", showLogin);
}

function showCompleteProfile() {
  showLanding();
  document.getElementById("authTabs")?.classList.add("d-none");
  document.querySelector(".auth-divider")?.classList.add("d-none");
  document.getElementById("btnGoogleLogin")?.classList.add("d-none");
  document.querySelector(".tab-content")?.classList.add("d-none");
  document.getElementById("completeProfileForm")?.classList.remove("d-none");

  const metadata = currentSession?.user?.user_metadata || {};
  const names = String(metadata.full_name || metadata.name || "")
    .trim()
    .split(/\s+/);

  const firstName =
    currentProfile?.first_name || metadata.first_name || names[0] || "";
  const lastName =
    currentProfile?.last_name ||
    metadata.last_name ||
    (names.length > 1 ? names.slice(1).join(" ") : "");

  document.getElementById("profileFirstName").value = firstName;
  document.getElementById("profileLastName").value = lastName;
  document.getElementById("profileSubject").value =
    currentProfile?.teaching_subject || "";
  document.getElementById("profilePhone").value = currentProfile?.phone || "";

  setAuthMessage("Complete your teacher profile to continue.", "info");
}

async function showApp() {
  document.getElementById("landingPage")?.classList.add("d-none");
  document.getElementById("appShell")?.classList.remove("d-none");
  document.getElementById("appAcknowledgment")?.classList.remove("d-none");

  const user = currentSession?.user;
  const userBadge = document.getElementById("userBadge");
  if (userBadge) {
    const fullName = [currentProfile?.first_name, currentProfile?.last_name]
      .filter(Boolean)
      .join(" ");
    userBadge.textContent = fullName || user?.email || "";
  }

  await populateNetworksSelect();
  startTutorialIfNeeded();
}

async function routeAuthenticatedUser() {
  if (!currentSession?.user) {
    showAuthForms();
    showLanding();
    return;
  }

  await loadProfile();

  if (!currentProfile) {
    const pending = localStorage.getItem("neurobuilder-pending-profile");
    if (pending) {
      try {
        const parsed = JSON.parse(pending);
        if (parsed.first_name && parsed.last_name && parsed.teaching_subject) {
          await upsertProfile(parsed);
          localStorage.removeItem("neurobuilder-pending-profile");
        }
      } catch (error) {
        console.warn("[Profile] Pending profile ignored:", error);
      }
    }
  }

  if (hasCompleteProfile(currentProfile)) {
    setAuthMessage("");
    await showApp();
  } else {
    showCompleteProfile();
  }
}

function profilePayloadFromSignup() {
  return {
    first_name: document.getElementById("signupFirstName")?.value.trim() || "",
    last_name: document.getElementById("signupLastName")?.value.trim() || "",
    teaching_subject:
      document.getElementById("signupSubject")?.value.trim() || "",
    phone: document.getElementById("signupPhone")?.value.trim() || null,
  };
}

function profilePayloadFromCompletion() {
  return {
    first_name: document.getElementById("profileFirstName")?.value.trim() || "",
    last_name: document.getElementById("profileLastName")?.value.trim() || "",
    teaching_subject:
      document.getElementById("profileSubject")?.value.trim() || "",
    phone: document.getElementById("profilePhone")?.value.trim() || null,
  };
}

async function loadProfile() {
  if (!supabaseClient || !currentSession?.user) return null;

  const { data, error } = await supabaseClient
    .from("profiles")
    .select("*")
    .eq("id", currentSession.user.id)
    .maybeSingle();

  if (error) {
    console.warn("[Profile] Load failed:", error);
    return null;
  }

  currentProfile = data;
  return currentProfile;
}

async function upsertProfile(profileFields) {
  if (!supabaseClient) throw new Error("Supabase client not available.");
  const session = await requireSession();

  const payload = {
    id: session.user.id,
    email: session.user.email,
    ...profileFields,
    updated_at: new Date().toISOString(),
  };

  const { data, error } = await supabaseClient
    .from("profiles")
    .upsert(payload, { onConflict: "id" })
    .select()
    .single();

  if (error) throw error;

  currentProfile = data;
  return data;
}

async function handleSignup(e) {
  e.preventDefault();

  const email = document.getElementById("signupEmail")?.value.trim();
  const password = document.getElementById("signupPassword")?.value || "";
  const profile = profilePayloadFromSignup();

  if (!email || !password || !profile.first_name || !profile.last_name) {
    setAuthMessage("Please fill in the required fields.", "error");
    return;
  }

  setAuthBusy(true);
  setAuthMessage("Creating your account...");

  try {
    const { data, error } = await supabaseClient.auth.signUp({
      email,
      password,
      options: {
        data: profile,
      },
    });

    if (error) throw error;

    currentSession = data.session;

    if (currentSession?.user) {
      await upsertProfile(profile);
      await routeAuthenticatedUser();
    } else {
      setAuthMessage(
        "Account created. Check your email to confirm it, then log in.",
        "success",
      );
    }
  } catch (error) {
    console.error("[Auth] Signup failed:", error);
    setAuthMessage(error.message || "Signup failed.", "error");
  } finally {
    setAuthBusy(false);
  }
}

async function handleLogin(e) {
  e.preventDefault();

  const email = document.getElementById("loginEmail")?.value.trim();
  const password = document.getElementById("loginPassword")?.value || "";

  setAuthBusy(true);
  setAuthMessage("Logging in...");

  try {
    const { data, error } = await supabaseClient.auth.signInWithPassword({
      email,
      password,
    });

    if (error) throw error;

    currentSession = data.session;
    await routeAuthenticatedUser();
  } catch (error) {
    console.error("[Auth] Login failed:", error);
    setAuthMessage(error.message || "Login failed.", "error");
  } finally {
    setAuthBusy(false);
  }
}

async function handleGoogleLogin() {
  const profile = profilePayloadFromSignup();
  if (profile.first_name || profile.last_name || profile.teaching_subject) {
    localStorage.setItem("neurobuilder-pending-profile", JSON.stringify(profile));
  }

  setAuthMessage("Redirecting to Google...");

  const { error } = await supabaseClient.auth.signInWithOAuth({
    provider: "google",
    options: {
      redirectTo: "https://micheleminno.github.io/neural-network-playground/",
    },
  });

  if (error) {
    console.error("[Auth] Google login failed:", error);
    setAuthMessage(error.message || "Google login failed.", "error");
  }
}

async function handleCompleteProfile(e) {
  e.preventDefault();

  const profile = profilePayloadFromCompletion();

  if (!profile.first_name || !profile.last_name || !profile.teaching_subject) {
    setAuthMessage("Please fill in the required fields.", "error");
    return;
  }

  setAuthBusy(true);

  try {
    await upsertProfile(profile);
    localStorage.removeItem("neurobuilder-pending-profile");
    setAuthMessage("");
    await showApp();
  } catch (error) {
    console.error("[Profile] Completion failed:", error);
    setAuthMessage(error.message || "Profile save failed.", "error");
  } finally {
    setAuthBusy(false);
  }
}

async function handleLogout() {
  if (!supabaseClient) return;

  await supabaseClient.auth.signOut();
  currentSession = null;
  currentProfile = null;
  currentNetworkId = null;
  currentNetworkName = "";

  document.getElementById("savedNetworks").value = "";
  document.getElementById("savedNetworkLabel").textContent =
    "-- Select network --";
  document.getElementById("btnUpdateNetwork")?.setAttribute("disabled", "");
  updateNetworkTitle();
  showAuthForms();
  activateAuthTab("login");
  showLanding();
}

function bindAuthUI() {
  document.getElementById("signupForm")?.addEventListener("submit", handleSignup);
  document.getElementById("loginForm")?.addEventListener("submit", handleLogin);
  document
    .getElementById("btnGoogleLogin")
    ?.addEventListener("click", handleGoogleLogin);
  document
    .getElementById("completeProfileForm")
    ?.addEventListener("submit", handleCompleteProfile);
  document.getElementById("btnLogout")?.addEventListener("click", handleLogout);
}

async function initAuthUI() {
  if (!supabaseClient) {
    setAuthMessage("Supabase client is not available.", "error");
    showLanding();
    return;
  }

  bindAuthUI();

  currentSession = await getSession();
  if (currentSession?.user) {
    await routeAuthenticatedUser();
  } else {
    showAuthForms();
    showLanding();
  }

  supabaseClient.auth.onAuthStateChange(async (_event, session) => {
    currentSession = session;

    if (session?.user) {
      await routeAuthenticatedUser();
    } else {
      currentProfile = null;
      showAuthForms();
      showLanding();
    }
  });
}
