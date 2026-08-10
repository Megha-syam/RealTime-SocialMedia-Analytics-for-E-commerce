export default function StatCard({ title, value, hint }) {
  return (
    <div className="glass card">
      <div className="muted">{title}</div>
      <div className="metric">{value}</div>
      {hint ? <div className="muted">{hint}</div> : null}
    </div>
  );
}
