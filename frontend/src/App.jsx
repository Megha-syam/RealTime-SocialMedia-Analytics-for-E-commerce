import { Navigate, Route, Routes } from "react-router-dom";
import Navbar from "./components/Navbar";
import ComparisonPage from "./pages/ComparisonPage";
import DashboardPage from "./pages/DashboardPage";
import LoginPage from "./pages/LoginPage";
import ProfilePage from "./pages/ProfilePage";
import SearchPage from "./pages/SearchPage";
import SummaryPage from "./pages/SummaryPage";

function PrivateLayout({ children }) {
  const token = localStorage.getItem("rse_token");
  if (!token) return <Navigate to="/login" replace />;

  return (
    <div className="app-shell">
      <Navbar />
      {children}
    </div>
  );
}

export default function App() {
  return (
    <Routes>
      <Route path="/login" element={<LoginPage />} />
      <Route
        path="/profile"
        element={
          <PrivateLayout>
            <ProfilePage />
          </PrivateLayout>
        }
      />
      <Route
        path="/profile/*"
        element={
          <PrivateLayout>
            <ProfilePage />
          </PrivateLayout>
        }
      />
      <Route
        path="/search"
        element={
          <PrivateLayout>
            <SearchPage />
          </PrivateLayout>
        }
      />
      <Route
        path="/dashboard"
        element={
          <PrivateLayout>
            <DashboardPage />
          </PrivateLayout>
        }
      />
      <Route
        path="/compare"
        element={
          <PrivateLayout>
            <ComparisonPage />
          </PrivateLayout>
        }
      />
      <Route
        path="/summary"
        element={
          <PrivateLayout>
            <SummaryPage />
          </PrivateLayout>
        }
      />
      <Route path="*" element={<Navigate to="/search" replace />} />
    </Routes>
  );
}
