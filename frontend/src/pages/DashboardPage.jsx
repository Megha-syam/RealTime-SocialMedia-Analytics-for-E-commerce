import { useEffect, useMemo, useState } from "react";
import { io } from "socket.io-client";
import {
  Cell,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { getDashboard } from "../api/client";
import StatCard from "../components/StatCard";

const socketBase = import.meta.env.VITE_SOCKET_URL || "http://127.0.0.1:5000";
const sentimentColors = {
  Positive: "#27c38f",
  Neutral: "#f6c445",
  Negative: "#ef5b6a",
};

export default function DashboardPage() {
  const [slug, setSlug] = useState(localStorage.getItem("rse_last_slug") || "iphone-15");
  const [dashboard, setDashboard] = useState(null);
  const [loading, setLoading] = useState(false);
  const [streamEvent, setStreamEvent] = useState(null);

  useEffect(() => {
    loadDashboard(slug);
  }, [slug]);

  useEffect(() => {
    const socket = io(socketBase, { path: "/socket.io", transports: ["websocket"] });
    socket.on("connect", () => {
      socket.emit("subscribe_product", { product: slug });
    });
    socket.on("analytics_update", (payload) => {
      setStreamEvent(payload);
      if (payload.slug === slug) {
        loadDashboard(slug);
      }
    });
    return () => socket.disconnect();
  }, [slug]);

  async function loadDashboard(currentSlug) {
    setLoading(true);
    try {
      const data = await getDashboard(currentSlug);
      setDashboard(data);
    } finally {
      setLoading(false);
    }
  }

  const sentimentData = useMemo(() => {
    if (!dashboard) return [];
    return [
      { name: "Positive", value: dashboard.sentiment_distribution.positive },
      { name: "Neutral", value: dashboard.sentiment_distribution.neutral },
      { name: "Negative", value: dashboard.sentiment_distribution.negative },
    ];
  }, [dashboard]);

  const timeline = dashboard?.timeline || [];
  const totalMentions = timeline.reduce((acc, row) => acc + row.mentions, 0);
  const avgTrend = timeline.length ? (timeline.reduce((acc, row) => acc + row.trend_score, 0) / timeline.length).toFixed(1) : "0";
  const bertAccuracy = dashboard?.model_metrics?.bert?.accuracy;
  const lstmConfidence = dashboard?.model_metrics?.lstm?.confidence;

  return (
    <div>
      <section className="glass hero">
        <h1 className="title">Live Intelligence Dashboard</h1>
        <p className="subtitle">Realtime updates via WebSocket, dynamic trend curves, sentiment mix, and risk alerts.</p>
      </section>

      <div className="glass card" style={{ marginBottom: "1rem" }}>
        <label className="muted">Product Slug</label>
        <div style={{ display: "flex", gap: "0.8rem", marginTop: "0.6rem", flexWrap: "wrap" }}>
          <input className="input" value={slug} onChange={(e) => setSlug(e.target.value)} />
          <button className="button button-secondary" style={{ width: "180px" }} onClick={() => loadDashboard(slug)}>
            Refresh
          </button>
        </div>
        {streamEvent ? (
          <div className="muted" style={{ marginTop: "0.7rem" }}>
            Live event: {streamEvent.product} | Forecast mentions {streamEvent.forecast_mentions} | Sentiment {streamEvent.forecast_sentiment}
          </div>
        ) : null}
        {dashboard?.product?.category ? (
          <div className="muted" style={{ marginTop: "0.4rem" }}>
            Category: {dashboard.product.category}
          </div>
        ) : null}
      </div>

      <section className="grid">
        <div className="col-4">
          <StatCard title="Total Mentions" value={totalMentions} hint={dashboard?.product?.name || "No product"} />
        </div>
        <div className="col-4">
          <StatCard title="Avg Trend Score" value={avgTrend} hint="Computed from time windows" />
        </div>
        <div className="col-4">
          <StatCard title="Realtime" value={loading ? "Syncing" : "Connected"} hint="Socket stream active" />
        </div>
        <div className="col-4">
          <StatCard
            title="BERT Accuracy"
            value={bertAccuracy != null ? `${(Number(bertAccuracy) * 100).toFixed(1)}%` : "N/A"}
            hint="Validation accuracy"
          />
        </div>
        <div className="col-4">
          <StatCard
            title="LSTM Confidence"
            value={lstmConfidence != null ? `${(Number(lstmConfidence) * 100).toFixed(1)}%` : "N/A"}
            hint="Forecast confidence"
          />
        </div>

        <div className="col-8">
          <div className="glass card" style={{ height: "350px" }}>
            <h3>Sentiment + Mention Timeline</h3>
            <ResponsiveContainer width="100%" height="88%">
              <LineChart data={timeline}>
                <CartesianGrid strokeDasharray="4 4" stroke="rgba(255,255,255,0.15)" />
                <XAxis dataKey="ts" hide />
                <YAxis yAxisId="left" />
                <YAxis yAxisId="right" orientation="right" />
                <Tooltip />
                <Legend />
                <Line yAxisId="left" type="monotone" dataKey="mentions" stroke="#12d2b5" strokeWidth={2} />
                <Line yAxisId="right" type="monotone" dataKey="avg_sentiment" stroke="#f9b54c" strokeWidth={2} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="col-4">
          <div className="glass card" style={{ height: "350px" }}>
            <h3>Sentiment Mix</h3>
            <ResponsiveContainer width="100%" height="88%">
              <PieChart>
                <Pie data={sentimentData} dataKey="value" nameKey="name" outerRadius={95}>
                  {sentimentData.map((entry) => (
                    <Cell key={entry.name} fill={sentimentColors[entry.name] || "#12d2b5"} />
                  ))}
                </Pie>
                <Tooltip />
                <Legend />
              </PieChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="col-12">
          {dashboard?.ai_dashboard_insight ? (
            <div className="glass card" style={{ marginBottom: "1rem" }}>
              <h3>AI Insight</h3>
              <p>{dashboard.ai_dashboard_insight}</p>
            </div>
          ) : null}
        </div>

        <div className="col-12">
          <div className="glass card">
            <h3>Risk Alerts</h3>
            <ul className="list-reset">
              {(dashboard?.risk_events || []).map((r, idx) => (
                <li key={idx} className="list-item">
                  <span className={`severity-${r.severity}`}>{r.severity.toUpperCase()}</span> | {r.trigger} | {r.details}
                </li>
              ))}
              {(dashboard?.risk_events || []).length === 0 ? <li className="list-item muted">No risks detected yet.</li> : null}
            </ul>
          </div>
        </div>
      </section>
    </div>
  );
}
