import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { login, register } from "../api/client";

export default function LoginPage() {
  const [mode, setMode] = useState("login");
  const [form, setForm] = useState({ full_name: "", email: "", password: "" });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const navigate = useNavigate();

  const onChange = (event) => {
    setForm((prev) => ({ ...prev, [event.target.name]: event.target.value }));
  };

  const onSubmit = async (event) => {
    event.preventDefault();
    setLoading(true);
    setError("");
    try {
      if (mode === "register") {
        await register(form);
        setMode("login");
      } else {
        await login({ email: form.email, password: form.password });
        navigate("/profile");
      }
    } catch (err) {
      setError(err?.response?.data?.error || "Authentication failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-shell" style={{ paddingTop: "4rem", maxWidth: "720px" }}>
      <section className="glass hero">
        <h1 className="title">Real Time E-Commerce Intelligence</h1>
        <p className="subtitle">
          Live social media ingestion, BERT sentiment understanding, trend forecasting, and risk-aware decision analytics.
        </p>
      </section>

      <section className="glass card">
        <div style={{ display: "flex", gap: "0.7rem", marginBottom: "1rem" }}>
          <button className={`button ${mode === "login" ? "" : "button-secondary"}`} onClick={() => setMode("login")}>
            Login
          </button>
          <button className={`button ${mode === "register" ? "" : "button-secondary"}`} onClick={() => setMode("register")}>
            Register
          </button>
        </div>
        <form onSubmit={onSubmit}>
          {mode === "register" ? (
            <div style={{ marginBottom: "0.8rem" }}>
              <label className="muted">Full Name</label>
              <input className="input" name="full_name" value={form.full_name} onChange={onChange} required />
            </div>
          ) : null}
          <div style={{ marginBottom: "0.8rem" }}>
            <label className="muted">Email</label>
            <input className="input" name="email" type="email" value={form.email} onChange={onChange} required />
          </div>
          <div style={{ marginBottom: "0.8rem" }}>
            <label className="muted">Password</label>
            <input className="input" name="password" type="password" value={form.password} onChange={onChange} required />
          </div>
          {error ? <div className="severity-high">{error}</div> : null}
          <button className="button" disabled={loading} style={{ marginTop: "0.8rem" }}>
            {loading ? "Processing..." : mode === "login" ? "Sign In" : "Create Account"}
          </button>
        </form>
      </section>
    </div>
  );
}
