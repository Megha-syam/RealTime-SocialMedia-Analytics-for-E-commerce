import { useState } from "react";
import { getSummary } from "../api/client";

export default function SummaryPage() {
  const [slug, setSlug] = useState(localStorage.getItem("rse_last_slug") || "iphone-15");
  const [windowMinutes, setWindowMinutes] = useState(43200);
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);

  const loadSummary = async (event) => {
    event.preventDefault();
    setLoading(true);
    try {
      const result = await getSummary(slug, windowMinutes);
      setData(result);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <section className="glass hero">
        <h1 className="title">AI Review Summary</h1>
        <p className="subtitle">Condenses large social streams into executive-level insight: pros, complaints, and overall opinion.</p>
      </section>

      <section className="glass card" style={{ marginBottom: "1rem" }}>
        <form onSubmit={loadSummary} style={{ display: "flex", gap: "0.8rem" }}>
          <input className="input" value={slug} onChange={(e) => setSlug(e.target.value)} />
          <select className="select" value={windowMinutes} onChange={(e) => setWindowMinutes(Number(e.target.value))} style={{ width: "180px" }}>
            <option value={60}>Last 1 hour</option>
            <option value={180}>Last 3 hours</option>
            <option value={240}>Last 4 hours</option>
            <option value={720}>Last 12 hours</option>
            <option value={1440}>Last 1 day</option>
            <option value={10080}>Last 7 days</option>
            <option value={43200}>Last 30 days (1 month)</option>
          </select>
          <button className="button button-secondary" style={{ width: "210px" }} disabled={loading}>
            {loading ? "Generating..." : "Generate Summary"}
          </button>
        </form>
      </section>

      {data ? (
        <section className="grid">
          <div className="col-12">
            <div className="glass card">
              <h3>{data.product} | Overall Opinion</h3>
              <p>{data.summary.overall}</p>
              <p>{data.summary.recommendation_paragraph}</p>
              <p className="muted">Category: {data.category || "other"}</p>
              <p className="muted">Analyzed samples: {data.summary.sample_size}</p>
              <p className="muted">Live window: last {data.summary.window_minutes} minutes</p>
              <p className="muted">Signal quality: {data.summary.signal_quality}</p>
              <p className="muted">
                Sources: {Object.entries(data.summary.source_breakdown || {}).map(([k, v]) => `${k}(${v})`).join(", ")}
              </p>
              <p className="muted">Summary engine: {data.summary.ai_model || "deterministic"}</p>
            </div>
          </div>
          <div className="col-6">
            <div className="glass card">
              <h3>Top Positive Points</h3>
              <p>{data.summary.pros_paragraph || data.summary.pros.join(", ")}</p>
            </div>
          </div>
          <div className="col-6">
            <div className="glass card">
              <h3>Top Negative Issues</h3>
              <p>{data.summary.cons_paragraph || data.summary.cons.join(", ")}</p>
            </div>
          </div>
        </section>
      ) : null}
    </div>
  );
}
