import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { getTrending, searchProduct } from "../api/client";
import StatCard from "../components/StatCard";

export default function SearchPage() {
  const [query, setQuery] = useState("");
  const [trending, setTrending] = useState([]);
  const [loading, setLoading] = useState(false);
  const [status, setStatus] = useState("");
  const [trendCategory, setTrendCategory] = useState("");
  const navigate = useNavigate();

  useEffect(() => {
    refreshTrending();
  }, [trendCategory]);

  async function refreshTrending() {
    try {
      const rows = await getTrending(10, {
        category: trendCategory || undefined,
      });
      setTrending(rows);
    } catch {
      setTrending([]);
    }
  }

  const submit = async (event) => {
    event.preventDefault();
    if (!query.trim()) return;
    setLoading(true);
    setStatus("Collecting live data and running analytics...");
    try {
      const result = await searchProduct(query.trim());
      localStorage.setItem("rse_last_slug", result.slug);
      const sourceInfo = Object.entries(result.source_counts || {})
        .map(([k, v]) => `${k}:${v}`)
        .join(", ");
      const warning = result.warning ? ` ${result.warning}` : "";
      const aiLine = result.ai_search_insight ? ` AI: ${result.ai_search_insight}` : "";
      setStatus(
        `Live update completed: fetched ${result.fetched_posts}, ingested ${result.inserted_posts}, cleaned ${result.removed_noisy_posts}. Category: ${result.category || "other"}. Sources: ${sourceInfo}.${warning}${aiLine}`
      );
      await refreshTrending();
      navigate("/dashboard");
    } catch (error) {
      setStatus(error?.response?.data?.error || "Search failed.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <section className="glass hero">
        <h1 className="title">Realtime Product Search</h1>
        <p className="subtitle">
          Pulls live social conversations, scores contextual sentiment, and computes trend signals instantly.
        </p>
      </section>

      <section className="grid">
        <div className="col-8">
          <div className="glass card">
            <form onSubmit={submit}>
              <label className="muted">Search Product</label>
              <div style={{ display: "flex", gap: "0.8rem", marginTop: "0.6rem" }}>
                <input
                  className="input"
                  placeholder="iPhone 15, Samsung S24..."
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                />
                <button className="button" disabled={loading} style={{ width: "220px" }}>
                  {loading ? "Analyzing..." : "Run Live Analytics"}
                </button>
              </div>
              {status ? <div style={{ marginTop: "0.8rem" }} className="muted">{status}</div> : null}
            </form>
          </div>
        </div>
        <div className="col-4">
          <StatCard title="Tracking Sources" value="Reddit + X + News + Trends" hint="Gemini is used for AI insights and category tagging." />
        </div>

        <div className="col-12">
          <div className="glass card">
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: "0.6rem", flexWrap: "wrap" }}>
              <h3 style={{ margin: 0 }}>Top Trending Products</h3>
              <div style={{ display: "flex", gap: "0.5rem", alignItems: "center", flexWrap: "wrap" }}>
                <select className="select" value={trendCategory} onChange={(e) => setTrendCategory(e.target.value)} style={{ width: "170px" }}>
                  <option value="">All Categories</option>
                  <option value="smartphone">Smartphone</option>
                  <option value="laptop">Laptop</option>
                  <option value="tablet">Tablet</option>
                  <option value="tv">TV</option>
                  <option value="wearable">Wearable</option>
                  <option value="automotive">Automotive</option>
                  <option value="bike">Bike</option>
                  <option value="appliance">Appliance</option>
                  <option value="gaming">Gaming</option>
                  <option value="other">Other</option>
                </select>
                <button className="button button-secondary" style={{ width: "180px" }} onClick={refreshTrending}>
                  Refresh Trends
                </button>
              </div>
            </div>
            <ul className="list-reset">
              {trending.map((item) => (
                <li key={item.slug} className="list-item" style={{ display: "grid", gridTemplateColumns: "2fr 1fr 1fr 1fr 1fr" }}>
                  <span>{item.product}</span>
                  <span className="muted">{item.category || "other"}</span>
                  <span className="muted">Trend {item.trend_score}</span>
                  <span className="muted">Sent {item.avg_sentiment}</span>
                  <span className="muted">Mentions {item.mentions}</span>
                </li>
              ))}
              {trending.length === 0 ? <li className="list-item muted">No trend data yet. Run a product search.</li> : null}
            </ul>
          </div>
        </div>
      </section>
    </div>
  );
}
