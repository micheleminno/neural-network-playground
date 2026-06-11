const SUPABASE_URL = "https://ajdnlxgbkkwihqiflyby.supabase.co";

const SUPABASE_KEY = "sb_publishable_CmojWuYCYnwEh0kgfZ5fmw_DfsQ7HrE";

async function saveNetwork() {
  const payload = {
    name: prompt("Nome rete?"),

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

  if (!data.length) return;

  const network = data[0];

  console.log(network);

  // qui poi ricostruiremo la rete
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
