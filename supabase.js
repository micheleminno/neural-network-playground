const SUPABASE_URL = "https://ajdnlxgbkkwihqiflyby.supabase.co";

const SUPABASE_KEY = "sb_publishable_CmojWuYCYnwEh0kgfZ5fmw_DfsQ7HrE";

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
    }),
  });

  console.log("STATUS", res.status);

  const txt = await res.text();
  console.log(txt);
}
