import axios from "axios";

const api = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL || "http://localhost:5000/api/v1",
  timeout: 30000,
});

api.interceptors.request.use((config) => {
  const token = localStorage.getItem("rse_token");
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

api.interceptors.response.use(
  (response) => response,
  (error) => {
    const status = error?.response?.status;
    const msg = error?.response?.data?.msg || error?.response?.data?.error || "";

    // Handle stale/invalid JWTs cleanly in UI.
    if (status === 401 || status === 422 || msg.includes("Subject must be a string")) {
      localStorage.removeItem("rse_token");
      localStorage.removeItem("rse_user");
      localStorage.removeItem("rse_last_slug");
      if (!window.location.pathname.includes("/login")) {
        window.location.href = "/login";
      }
    }

    return Promise.reject(error);
  }
);

export async function register(payload) {
  const { data } = await api.post("/auth/register", payload);
  return data;
}

export async function login(payload) {
  const { data } = await api.post("/auth/login", payload);
  localStorage.setItem("rse_token", data.access_token);
  localStorage.setItem("rse_user", JSON.stringify(data.user));
  return data;
}

export async function getProfile() {
  const { data } = await api.get("/auth/profile");
  return data;
}

export async function updateProfile(fullName) {
  const { data } = await api.put("/auth/profile", { full_name: fullName });
  const existing = JSON.parse(localStorage.getItem("rse_user") || "{}");
  localStorage.setItem("rse_user", JSON.stringify({ ...existing, ...data }));
  return data;
}

export async function changePassword(currentPassword, newPassword) {
  const { data } = await api.post("/auth/change-password", {
    current_password: currentPassword,
    new_password: newPassword,
  });
  return data;
}

export async function searchProduct(query) {
  const { data } = await api.post("/products/search", { query });
  return data;
}

export async function getDashboard(slug, options = {}) {
  const params = new URLSearchParams();
  if (options.windowHours) params.set("window_hours", String(options.windowHours));
  if (options.year) params.set("year", String(options.year));
  if (options.month) params.set("month", String(options.month));
  const suffix = params.toString() ? `?${params.toString()}` : "";
  const { data } = await api.get(`/products/${slug}/dashboard${suffix}`);
  return data;
}

export async function getSummary(slug, windowMinutes = 43200, options = {}) {
  const params = new URLSearchParams({ window_minutes: String(windowMinutes) });
  if (options.year) params.set("year", String(options.year));
  if (options.month) params.set("month", String(options.month));
  const { data } = await api.get(`/products/${slug}/summary?${params.toString()}`);
  return data;
}

export async function compareProducts(left, right, windowMinutes = 43200, options = {}) {
  const { data } = await api.post("/products/compare", {
    left,
    right,
    window_minutes: windowMinutes,
    year: options.year || null,
    month: options.month || null,
  });
  return data;
}

export async function getTrending(limit = 10, options = {}) {
  const params = new URLSearchParams({ limit: String(limit) });
  if (options.category) params.set("category", options.category);
  if (options.year) params.set("year", String(options.year));
  if (options.month) params.set("month", String(options.month));
  const { data } = await api.get(`/products/trending?${params.toString()}`);
  return data.items;
}

export default api;
