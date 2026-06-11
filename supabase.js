const SUPABASE_URL = "https://ajdnlxgbkkwihqiflyby.supabase.co";

const SUPABASE_KEY = "sb_publishable_CmojWuYCYnwEh0kgfZ5fmw_DfsQ7HrE";

async function saveNetwork() {
  const networkName = prompt("Network name?");
  if (!networkName) return;

  currentNetworkName = networkName;
  updateNetworkTitle();

  const payload = {
    name: networkName,

    architecture: arch,

    weights: net.layers.map((layer) => ({
      W: layer.W,
      b: layer.b,
      activation: layer.activation,
      useBias: layer.useBias,
    })),

    dataset: {
      X: dataset.X,
      y: dataset.y,
    },

    loss: Number($("#lossNow")?.textContent || 0),

    accuracy: parseFloat(($("#accNow")?.textContent || "0").replace("%", "")),
  };

  const r = await fetch(`${SUPABASE_URL}/rest/v1/networks`, {
    method: "POST",
    headers: {
      apikey: SUPABASE_KEY,
      Authorization: `Bearer ${SUPABASE_KEY}`,
      "Content-Type": "application/json",
      Prefer: "return=representation",
    },
    body: JSON.stringify(payload),
  });

  const txt = await r.text();

  console.log("STATUS", r.status);
  console.log(txt);
}

async function loadNetworks() {
  const r = await fetch(
    `${SUPABASE_URL}/rest/v1/networks?select=id,name,created_at`,
    {
      headers: {
        apikey: SUPABASE_KEY,
        Authorization: `Bearer ${SUPABASE_KEY}`,
      },
    },
  );

  const data = await r.json();

  console.log("NETWORKS:", data);

  return data;
}

async function populateNetworksSelect() {
  console.log("LOADING NETWORKS...");

  const networks = await loadNetworks();

  const sel = document.getElementById("savedNetworks");

  if (!sel) return;

  sel.innerHTML = '<option value="">-- Select network --</option>';

  networks.forEach((n) => {
    const opt = document.createElement("option");

    opt.value = n.id;

    opt.textContent = n.name;

    sel.appendChild(opt);
  });

  console.log("SELECT POPULATED");
}

async function loadNetworkById(id) {
  const r = await fetch(
    `${SUPABASE_URL}/rest/v1/networks?id=eq.${id}&select=*`,
    {
      headers: {
        apikey: SUPABASE_KEY,
        Authorization: `Bearer ${SUPABASE_KEY}`,
      },
    },
  );

  const data = await r.json();

  if (!data.length) {
    alert("Network not found");
    return;
  }

  const network = data[0];
  currentNetworkName = network.name;
  updateNetworkTitle();

  // ==========================
  // ARCHITETTURA
  // ==========================

  arch = network.architecture.layers;
  console.log("Architecture:", arch);

  const inputLayer = arch.find((l) => l.type === "input");
  const outputLayer = arch.find((l) => l.type === "output");

  if (inputLayer) inputSize = inputLayer.neurons;
  if (outputLayer) outputSize = outputLayer.neurons;

  buildNetwork();

  // ==========================
  // PESI
  // ==========================

  if (network.weights && network.weights.length === net.layers.length) {
    for (let i = 0; i < net.layers.length; i++) {
      net.layers[i].W = network.weights[i].W;

      net.layers[i].b = network.weights[i].b;

      net.layers[i].activation = network.weights[i].activation;

      net.layers[i].useBias = network.weights[i].useBias;
    }
  }

  // ==========================
  // DATASET
  // ==========================

  if (network.dataset) {
    dataset.X = network.dataset.X || [];

    dataset.y = network.dataset.y || [];
  }

  // ==========================
  // METRICHE
  // ==========================

  if ($("#lossNow")) $("#lossNow").textContent = network.loss ?? "-";

  if ($("#accNow"))
    $("#accNow").textContent =
      network.accuracy != null ? network.accuracy + "%" : "-";

  renderArchitecture();
  renderTestInputs();
  renderNNVis();
  updateJSON();

  console.log("Network loaded:", network.name);
}
// TEST

async function testSupabase() {
  const res = await fetch(`${SUPABASE_URL}/rest/v1/networks?select=*`, {
    headers: {
      apikey: SUPABASE_KEY,
      Authorization: `Bearer ${SUPABASE_KEY}`,
    },
  });

  const data = await res.json();

  console.log("SUPABASE:", data);
}

async function saveTest() {
  const res = await fetch(`${SUPABASE_URL}/rest/v1/networks`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      apikey: SUPABASE_KEY,
      Authorization: `Bearer ${SUPABASE_KEY}`,
    },
    body: JSON.stringify({
      name: "Rete di prova",
      architecture: {
        layers: [2, 4, 1],
      },
      weights: {
        layers: [],
      },
      dataset: {
        name: "xor",
      },
      loss: null,
      accuracy: null,
    }),
  });

  console.log("STATUS", res.status);
  console.log(await res.text());
}
