const SUPABASE_URL = "https://ajdnlxgbkkwihqiflyby.supabase.co";

const SUPABASE_KEY = "sb_publishable_QUI_LA_TUA_CHIAVE";

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
