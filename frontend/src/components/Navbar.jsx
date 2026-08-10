import { Link, useLocation, useNavigate } from "react-router-dom";

export default function Navbar() {
  const location = useLocation();
  const navigate = useNavigate();

  const logout = () => {
    localStorage.removeItem("rse_token");
    localStorage.removeItem("rse_user");
    navigate("/login");
  };

  const links = [
    { to: "/search", label: "Search" },
    { to: "/dashboard", label: "Dashboard" },
    { to: "/compare", label: "Compare" },
    { to: "/summary", label: "AI Summary" },
    { to: "/profile", label: "Profile" },
  ];

  return (
    <header className="topbar glass">
      <div className="brand">
        <div className="brand-tag">RSE-LIVE</div>
        <span>Realtime E-Commerce Analytics</span>
      </div>
      <nav className="nav-links">
        {links.map((link) => (
          <Link
            key={link.to}
            to={link.to}
            className="nav-pill"
            style={{
              outline: location.pathname.startsWith(link.to)
                ? "1px solid rgba(18, 210, 181, 0.8)"
                : "none",
            }}
          >
            {link.label}
          </Link>
        ))}
        <button className="nav-pill" onClick={logout} style={{ border: "none", cursor: "pointer" }}>
          Logout
        </button>
      </nav>
    </header>
  );
}
