import { useEffect, useState } from "react";
import { changePassword, getProfile, updateProfile } from "../api/client";

export default function ProfilePage() {
  const [profile, setProfile] = useState(null);
  const [fullName, setFullName] = useState("");
  const [currentPassword, setCurrentPassword] = useState("");
  const [newPassword, setNewPassword] = useState("");
  const [loadingProfile, setLoadingProfile] = useState(false);
  const [savingProfile, setSavingProfile] = useState(false);
  const [savingPassword, setSavingPassword] = useState(false);
  const [profileMsg, setProfileMsg] = useState("");
  const [passwordMsg, setPasswordMsg] = useState("");
  const [error, setError] = useState("");

  useEffect(() => {
    loadProfile();
  }, []);

  async function loadProfile() {
    setLoadingProfile(true);
    setError("");
    try {
      const data = await getProfile();
      setProfile(data);
      setFullName(data.full_name || "");
    } catch (err) {
      setError(err?.response?.data?.error || "Failed to load profile");
    } finally {
      setLoadingProfile(false);
    }
  }

  async function onSaveProfile(event) {
    event.preventDefault();
    setSavingProfile(true);
    setProfileMsg("");
    setError("");
    try {
      const data = await updateProfile(fullName);
      setProfile(data);
      setProfileMsg("Profile updated successfully.");
    } catch (err) {
      setError(err?.response?.data?.error || "Failed to update profile");
    } finally {
      setSavingProfile(false);
    }
  }

  async function onChangePassword(event) {
    event.preventDefault();
    setSavingPassword(true);
    setPasswordMsg("");
    setError("");
    try {
      const data = await changePassword(currentPassword, newPassword);
      setPasswordMsg(data?.message || "Password updated.");
      setCurrentPassword("");
      setNewPassword("");
    } catch (err) {
      setError(err?.response?.data?.error || "Failed to change password");
    } finally {
      setSavingPassword(false);
    }
  }

  return (
    <div>
      <section className="glass hero">
        <h1 className="title">User Profile</h1>
        <p className="subtitle">Manage your account details and securely change your password.</p>
      </section>

      {loadingProfile ? (
        <section className="glass card">
          <p className="muted">Loading profile...</p>
        </section>
      ) : null}

      {error ? (
        <section className="glass card" style={{ marginBottom: "1rem" }}>
          <p className="severity-high">{error}</p>
        </section>
      ) : null}

      <section className="grid">
        <div className="col-6">
          <div className="glass card">
            <h3>Account Information</h3>
            <form onSubmit={onSaveProfile}>
              <div style={{ marginBottom: "0.8rem" }}>
                <label className="muted">Email</label>
                <input className="input" value={profile?.email || ""} disabled />
              </div>
              <div style={{ marginBottom: "0.8rem" }}>
                <label className="muted">Role</label>
                <input className="input" value={profile?.role || ""} disabled />
              </div>
              <div style={{ marginBottom: "0.8rem" }}>
                <label className="muted">Full Name</label>
                <input
                  className="input"
                  value={fullName}
                  onChange={(e) => setFullName(e.target.value)}
                  required
                />
              </div>
              {profileMsg ? <p className="severity-low">{profileMsg}</p> : null}
              <button className="button" disabled={savingProfile}>
                {savingProfile ? "Saving..." : "Save Profile"}
              </button>
            </form>
          </div>
        </div>

        <div className="col-6">
          <div className="glass card">
            <h3>Change Password</h3>
            <form onSubmit={onChangePassword}>
              <div style={{ marginBottom: "0.8rem" }}>
                <label className="muted">Current Password</label>
                <input
                  className="input"
                  type="password"
                  value={currentPassword}
                  onChange={(e) => setCurrentPassword(e.target.value)}
                  required
                />
              </div>
              <div style={{ marginBottom: "0.8rem" }}>
                <label className="muted">New Password</label>
                <input
                  className="input"
                  type="password"
                  value={newPassword}
                  onChange={(e) => setNewPassword(e.target.value)}
                  minLength={8}
                  required
                />
              </div>
              {passwordMsg ? <p className="severity-low">{passwordMsg}</p> : null}
              <button className="button button-secondary" disabled={savingPassword}>
                {savingPassword ? "Updating..." : "Change Password"}
              </button>
            </form>
          </div>
        </div>
      </section>
    </div>
  );
}
