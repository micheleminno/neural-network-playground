const SUPABASE_URL = "https://ajdnlxgbkkwihqiflyby.supabase.co";

const SUPABASE_KEY = "sb_publishable_CmojWuYCYnwEh0kgfZ5fmw_DfsQ7HrE";

async function saveNetwork() {
  const networkName = prompt("Network name?");
  if (!networkName) return;

  currentNetworkName = networkName;
  updateNetworkTitle();

  const payload = {
    name: networkName,

    architecture: {
      layers: arch,
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
  const networks = await loadNetworks();

  const menu = document.getElementById("savedNetworksMenu");
  const hidden = document.getElementById("savedNetworks");
  const label = document.getElementById("savedNetworkLabel");

  if (!menu || !hidden || !label) return;

  menu.innerHTML = "";

  networks.forEach((n) => {
    const li = document.createElement("li");

    li.innerHTML = `
      <button
        type="button"
        class="dropdown-item d-flex justify-content-between align-items-center saved-network-item"
        data-id="${n.id}"
        data-name="${n.name}"
      >
        <span>${n.name}</span>

        <span
          class="delete-network-btn opacity-0 text-danger fw-bold"
          data-id="${n.id}"
          style="cursor:pointer"
        >
          ×
        </span>
      </button>
    `;

    menu.appendChild(li);
  });

  menu.querySelectorAll(".saved-network-item").forEach((item) => {
    item.addEventListener("mouseenter", () => {
      item.querySelector(".delete-network-btn")?.classList.remove("opacity-0");
    });

    item.addEventListener("mouseleave", () => {
      item.querySelector(".delete-network-btn")?.classList.add("opacity-0");
    });

    item.addEventListener("click", () => {
      hidden.value = item.dataset.id;
      label.textContent = item.dataset.name;
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
  console.log("NETWORK", network);
  console.log("ARCHITECTURE", network.architecture);

  currentNetworkName = network.name;
  updateNetworkTitle();

  // ==========================
  // ARCHITETTURA
  // ==========================

  arch = network.architecture.layers;

  layerConfig = network.weights.map((w) => ({
    activation: w.activation,
    useBias: w.useBias,
  }));

  inputSize = arch[0];
  outputSize = arch[arch.length - 1];

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

async function deleteNetworkById(id) {
  if (!id) return;

  console.log("DELETE ID =", id);

  const ok = confirm("Delete this network from cloud?");
  if (!ok) return;

  const r = await fetch(`${SUPABASE_URL}/rest/v1/networks?id=eq.${id}`, {
    method: "DELETE",
    headers: {
      apikey: SUPABASE_KEY,
      Authorization: `Bearer ${SUPABASE_KEY}`,
    },
  });

  console.log("DELETE STATUS", r.status);

  if (!r.ok) {
    console.error(await r.text());
    alert("Delete failed");
    return;
  }

  document.getElementById("savedNetworks").value = "";
  document.getElementById("savedNetworkLabel").textContent =
    "-- Select network --";

  await populateNetworksSelect();
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
