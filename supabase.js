const SUPABASE_URL = "https://ajdnlxgbkkwihqiflyby.supabase.co";
const SUPABASE_KEY = "sb_publishable_CmojWuYCYnwEh0kgfZ5fmw_DfsQ7HrE";

const supabaseClient = window.supabase?.createClient(
  SUPABASE_URL,
  SUPABASE_KEY,
);

let currentSession = null;
let currentProfile = null;

async function getSession() {
  if (!supabaseClient) return null;

  const { data, error } = await supabaseClient.auth.getSession();
  if (error) {
    console.error("[Auth] Session error:", error);
    return null;
  }

  currentSession = data.session;
  return currentSession;
}

async function requireSession() {
  const session = currentSession || (await getSession());

  if (!session?.user) {
    throw new Error("Please log in first.");
  }

  return session;
}

async function authHeaders() {
  const session = await requireSession();

  return {
    apikey: SUPABASE_KEY,
    Authorization: `Bearer ${session.access_token}`,
  };
}

async function saveNetwork() {
  const session = await requireSession();
  const networkName = prompt("Network name?");
  if (!networkName) return;

  currentNetworkName = networkName;
  updateNetworkTitle();

  const payload = {
    user_id: session.user.id,
    name: networkName,

    architecture: {
      layers: arch,
      inputConfig: serializeInputConfig(),
    },

    weights: net.layers.map((layer) => ({
      W: layer.W,
      b: layer.b,
      activation: layer.activation,
      useBias: layer.useBias,
    })),

    dataset: {
      X: dataset.X,
      y: dataset.y,
      rawText: dataset.rawText,
    },

    loss: Number($("#lossNow")?.textContent || 0),

    accuracy: parseFloat(($("#accNow")?.textContent || "0").replace("%", "")),
  };

  const r = await fetch(`${SUPABASE_URL}/rest/v1/networks`, {
    method: "POST",
    headers: {
      ...(await authHeaders()),
      "Content-Type": "application/json",
      Prefer: "return=representation",
    },
    body: JSON.stringify(payload),
  });

  const txt = await r.text();
  console.log("STATUS", r.status);
  console.log(txt);

  if (!r.ok) {
    alert("Save failed");
    return;
  }

  const saved = JSON.parse(txt)[0];
  currentNetworkId = saved?.id || null;
  document.getElementById("btnUpdateNetwork")?.removeAttribute("disabled");
}

async function loadNetworks() {
  const session = await requireSession();
  const r = await fetch(
    `${SUPABASE_URL}/rest/v1/networks?select=id,name,created_at&user_id=eq.${session.user.id}&order=created_at.desc`,
    {
      headers: await authHeaders(),
    },
  );

  if (!r.ok) {
    console.error(await r.text());
    return [];
  }

  const data = await r.json();
  console.log("NETWORKS:", data);

  return data;
}

async function populateNetworksSelect() {
  const menu = document.getElementById("savedNetworksMenu");
  const hidden = document.getElementById("savedNetworks");
  const label = document.getElementById("savedNetworkLabel");

  if (!menu || !hidden || !label) return;

  menu.innerHTML = "";

  if (!currentSession?.user) {
    label.textContent = "-- Log in to load networks --";
    return;
  }

  const networks = await loadNetworks();

  if (!networks.length) {
    const li = document.createElement("li");
    const empty = document.createElement("span");
    empty.className = "dropdown-item text-secondary";
    empty.textContent = "No saved networks yet";
    li.appendChild(empty);
    menu.appendChild(li);
    label.textContent = "-- Select network --";
    return;
  }

  networks.forEach((n) => {
    const li = document.createElement("li");
    const item = document.createElement("div");
    const name = document.createElement("span");
    const btn = document.createElement("button");
    const icon = document.createElement("i");

    item.className =
      "dropdown-item d-flex justify-content-between align-items-center saved-network-item";
    item.dataset.id = n.id;
    item.dataset.name = n.name;
    item.style.cursor = "pointer";

    name.textContent = n.name;

    btn.type = "button";
    btn.className = "btn btn-danger btn-sm delete-network-btn opacity-0";
    btn.dataset.id = n.id;
    btn.title = "Delete network";
    btn.style.width = "32px";
    btn.style.height = "32px";
    btn.style.padding = "0";
    btn.style.display = "flex";
    btn.style.alignItems = "center";
    btn.style.justifyContent = "center";

    icon.className = "bi bi-x-lg";
    btn.appendChild(icon);
    item.appendChild(name);
    item.appendChild(btn);
    li.appendChild(item);
    menu.appendChild(li);
  });

  menu.querySelectorAll(".saved-network-item").forEach((item) => {
    item.addEventListener("mouseenter", () => {
      const btn = item.querySelector(".delete-network-btn");

      btn?.classList.remove("opacity-0");
      btn?.classList.add("opacity-100");
    });

    item.addEventListener("mouseleave", () => {
      const btn = item.querySelector(".delete-network-btn");

      btn?.classList.remove("opacity-100");
      btn?.classList.add("opacity-0");
    });
    item.addEventListener("click", async () => {
      hidden.value = item.dataset.id;
      label.textContent = item.dataset.name;

      await loadNetworkById(item.dataset.id);
    });
  });

  menu.querySelectorAll(".delete-network-btn").forEach((btn) => {
    btn.addEventListener("click", async (e) => {
      e.stopPropagation();

      await deleteNetworkById(btn.dataset.id);
    });
  });
}

async function loadNetworkById(id) {
  const session = await requireSession();
  const r = await fetch(
    `${SUPABASE_URL}/rest/v1/networks?id=eq.${id}&user_id=eq.${session.user.id}&select=*`,
    {
      headers: await authHeaders(),
    },
  );

  const data = await r.json();

  if (!data.length) {
    alert("Network not found");
    return;
  }

  const network = data[0];

  console.log("NETWORK", network);
  console.log("ARCHITECTURE", network.architecture);
  console.log("WEIGHTS ARRAY?", Array.isArray(network.weights));
  console.log("WEIGHTS", network.weights);

  currentNetworkName = network.name;
  currentNetworkId = network.id;
  document.getElementById("btnUpdateNetwork")?.removeAttribute("disabled");
  updateNetworkTitle();

  arch = network.architecture.layers.map((l) => ({
    id: l.id || crypto.randomUUID(),
    type: l.type,
    neurons: l.neurons,
    activation: l.activation,
    bias: l.bias,
  }));

  inputSize = arch.find((l) => l.type === "input")?.neurons || 1;
  outputSize = arch.find((l) => l.type === "output")?.neurons || 1;
  applyInputConfig(network.architecture.inputConfig, inputSize);

  if (inputConfig.mode === "text" && textFeatureCount() !== inputSize) {
    alert(t("textConfigMismatch"));
    return;
  }

  console.log("ARCH RAW", network.architecture.layers);
  console.log("ARCH MAPPED", arch);
  console.log(
    "ARCH TYPES",
    arch.map((x) => ({
      type: x.type,
      neurons: x.neurons,
    })),
  );

  buildNetwork();

  if (
    Array.isArray(network.weights) &&
    network.weights.length === net.layers.length
  ) {
    for (let i = 0; i < net.layers.length; i++) {
      net.layers[i].W = network.weights[i].W;
      net.layers[i].b = network.weights[i].b;
    }
  }

  if (network.dataset) {
    dataset.X = network.dataset.X || [];
    dataset.y = network.dataset.y || [];
    dataset.rawText = network.dataset.rawText || [];
    if (inputConfig.mode === "text" && dataset.rawText.length) {
      updateTextDatasetEncoding();
    }
  }

  if ($("#lossNow")) {
    $("#lossNow").textContent = network.loss ?? "-";
  }

  if ($("#accNow")) {
    $("#accNow").textContent =
      network.accuracy != null ? network.accuracy + "%" : "-";
  }

  renderArchitecture();
  renderTestInputs();
  renderNNVis();
  syncInputModeControls();
  updateJSON();

  console.log("Network loaded:", network.name);
}

async function updateNetwork(id) {
  if (!id) {
    alert("No network loaded");
    return;
  }

  const session = await requireSession();
  const payload = {
    user_id: session.user.id,

    architecture: {
      layers: arch,
      inputConfig: serializeInputConfig(),
    },

    weights: net.layers.map((layer) => ({
      W: layer.W,
      b: layer.b,
      activation: layer.activation,
      useBias: layer.useBias,
    })),

    dataset: {
      X: dataset.X,
      y: dataset.y,
      rawText: dataset.rawText,
    },

    loss: Number($("#lossNow")?.textContent || 0),

    accuracy: parseFloat(($("#accNow")?.textContent || "0").replace("%", "")),
  };

  const r = await fetch(
    `${SUPABASE_URL}/rest/v1/networks?id=eq.${id}&user_id=eq.${session.user.id}`,
    {
      method: "PATCH",
      headers: {
        ...(await authHeaders()),
        "Content-Type": "application/json",
        Prefer: "return=representation",
      },
      body: JSON.stringify(payload),
    },
  );

  const txt = await r.text();

  console.log("UPDATE STATUS", r.status);
  console.log(txt);

  if (r.ok) {
    alert("Network updated");
  }
}

async function deleteNetworkById(id) {
  if (!id) return;

  const session = await requireSession();
  console.log("DELETE ID =", id);

  const ok = confirm("Delete this network from cloud?");
  if (!ok) return;

  const r = await fetch(
    `${SUPABASE_URL}/rest/v1/networks?id=eq.${id}&user_id=eq.${session.user.id}`,
    {
      method: "DELETE",
      headers: await authHeaders(),
    },
  );

  console.log("DELETE STATUS", r.status);

  if (!r.ok) {
    console.error(await r.text());
    alert("Delete failed");
    return;
  }

  document.getElementById("savedNetworks").value = "";
  document.getElementById("savedNetworkLabel").textContent =
    "-- Select network --";

  if (currentNetworkId === id) {
    currentNetworkId = null;
    currentNetworkName = "";
    document.getElementById("btnUpdateNetwork")?.setAttribute("disabled", "");
    updateNetworkTitle();
  }

  await populateNetworksSelect();
}
