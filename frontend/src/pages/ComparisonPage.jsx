import { useState } from "react";
import { compareProducts } from "../api/client";

export default function ComparisonPage() {
  const [left, setLeft] = useState("iPhone 15");
  const [right, setRight] = useState("Samsung S24");
  const [windowMinutes, setWindowMinutes] = useState(43200);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const onCompare = async (event) => {
    event.preventDefault();
    setLoading(true);
    try {
      const data = await compareProducts(left, right, windowMinutes);
      setResult(data);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <section className="glass hero">
        <h1 className="title">AI Product Comparison</h1>
        <p className="subtitle">Run side-by-side social trend intelligence with sentiment, mentions, and recommendation engine.</p>
      </section>

      <section className="glass card" style={{ marginBottom: "1rem" }}>
        <form onSubmit={onCompare} style={{ display: "grid", gridTemplateColumns: "1fr 1fr 180px auto", gap: "0.8rem" }}>
          <input className="input" value={left} onChange={(e) => setLeft(e.target.value)} />
          <input className="input" value={right} onChange={(e) => setRight(e.target.value)} />
          <select className="select" value={windowMinutes} onChange={(e) => setWindowMinutes(Number(e.target.value))}>
            <option value={60}>Last 1 hour</option>
            <option value={180}>Last 3 hours</option>
            <option value={240}>Last 4 hours</option>
            <option value={720}>Last 12 hours</option>
            <option value={1440}>Last 1 day</option>
            <option value={10080}>Last 7 days</option>
            <option value={43200}>Last 30 days (1 month)</option>
          </select>
          <button className="button" style={{ width: "180px" }} disabled={loading}>
            {loading ? "Comparing..." : "Compare"}
          </button>
        </form>
      </section>

      {result ? (
        <section className="grid">
          <div className="col-12">
            <div className="glass card">
              <h3>Head-to-Head Comparison</h3>
              <p>{result.comparison_summary || result.recommendation_reason}</p>
              <p className="muted">
                Sentiment Delta (L-R): {result.deltas?.sentiment_delta_left_minus_right}
                {" | "}
                Trend Delta (L-R): {result.deltas?.trend_score_delta_left_minus_right}
                {" | "}
                Mentions Delta (L-R): {result.deltas?.mentions_delta_left_minus_right}
              </p>
              <p className="muted">
                Winner by Sentiment: {result.winner_by_dimension?.sentiment}
                {" | "}
                Winner by Trend: {result.winner_by_dimension?.trend_score}
                {" | "}
                Winner by Mentions: {result.winner_by_dimension?.mentions}
              </p>
            </div>
          </div>

          <div className="col-6">
            <div className="glass card">
              <h3>{result.left.product}</h3>
              <p className="muted">Category: {result.left.category || "other"}</p>
              <p className="muted">Sentiment: {result.left.sentiment}</p>
              <p className="muted">Trend Score: {result.left.trend_score}</p>
              <p className="muted">Mentions: {result.left.mentions}</p>
              <h4>Pros</h4>
              <p>{result.left.summary.pros_paragraph || result.left.summary.pros.join(", ")}</p>
              <h4>Cons</h4>
              <p>{result.left.summary.cons_paragraph || result.left.summary.cons.join(", ")}</p>
              <p className="muted">Window: last {result.left.summary.window_minutes} minutes</p>
            </div>
          </div>
          <div className="col-6">
            <div className="glass card">
              <h3>{result.right.product}</h3>
              <p className="muted">Category: {result.right.category || "other"}</p>
              <p className="muted">Sentiment: {result.right.sentiment}</p>
              <p className="muted">Trend Score: {result.right.trend_score}</p>
              <p className="muted">Mentions: {result.right.mentions}</p>
              <h4>Pros</h4>
              <p>{result.right.summary.pros_paragraph || result.right.summary.pros.join(", ")}</p>
              <h4>Cons</h4>
              <p>{result.right.summary.cons_paragraph || result.right.summary.cons.join(", ")}</p>
              <p className="muted">Window: last {result.right.summary.window_minutes} minutes</p>
            </div>
          </div>
          <div className="col-12">
            <div className="glass card">
              <h3>Recommendation</h3>
              <p>{result.recommendation_reason}</p>
            </div>
          </div>
        </section>
      ) : null}
    </div>
  );
}
