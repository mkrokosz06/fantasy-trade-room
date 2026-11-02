import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { BrowserRouter, NavLink, Route, Routes } from "react-router-dom";
import { useApi, type Meta, type Settings } from "./api";
import Advisor from "./pages/Advisor";
import LeaguePage from "./pages/LeaguePage";
import { EngineProvider } from "./engine";
import Matchups from "./pages/Matchups";
import ModelLab from "./pages/ModelLab";
import Player from "./pages/Player";
import Players from "./pages/Players";
import Rankings from "./pages/Rankings";
import SettingsPage from "./pages/SettingsPage";
import Trade from "./pages/Trade";
import "./styles.css";

function Shell() {
  const meta = useApi<Meta>("/api/meta");
  const settings = useApi<Settings>("/api/settings");
  return (
    <div className="shell">
      <nav className="side" aria-label="Main">
        <div className="brand">
          Trade Room
          <small>{settings.data ? `${settings.data.league.teams}-team, ${settings.data.scoring.label}` : "Fantasy football"}</small>
        </div>
        <NavLink to="/" end>Trade calculator</NavLink>
        <NavLink to="/advisor">Trade advisor</NavLink>
        <NavLink to="/rankings">Rankings</NavLink>
        <NavLink to="/players">Player stats</NavLink>
        <NavLink to="/matchups">Matchups</NavLink>
        <NavLink to="/league">ESPN league</NavLink>
        <NavLink to="/settings">Settings</NavLink>
        <div className="group">Model lab</div>
        <NavLink to="/model" end>Overview</NavLink>
        <NavLink to="/model/trade">Trade calculator</NavLink>
        <NavLink to="/model/advisor">Trade advisor</NavLink>
        <NavLink to="/model/rankings">Rankings</NavLink>
        <div className="foot">
          {meta.error
            ? "Backend not reachable. Is it running on port 8000?"
            : meta.data
              ? `Season ${meta.data.season}, stats through week ${meta.data.last_stats_week}`
              : "Loading data…"}
        </div>
      </nav>
      <main>
        <Routes>
          <Route path="/" element={<Trade />} />
          <Route path="/advisor" element={<Advisor />} />
          <Route path="/rankings" element={<Rankings />} />
          <Route path="/players" element={<Players />} />
          <Route path="/player/:id" element={<Player />} />
          <Route path="/matchups" element={<Matchups />} />
          <Route path="/league" element={<LeaguePage />} />
          <Route path="/settings" element={<SettingsPage />} />
          <Route path="/model" element={<ModelLab />} />
          <Route path="/model/trade" element={<EngineProvider engine="ml"><Trade /></EngineProvider>} />
          <Route path="/model/advisor" element={<EngineProvider engine="ml"><Advisor /></EngineProvider>} />
          <Route path="/model/rankings" element={<EngineProvider engine="ml"><Rankings /></EngineProvider>} />
          <Route path="/model/player/:id" element={<EngineProvider engine="ml"><Player /></EngineProvider>} />
          <Route path="*" element={<div className="empty">That page doesn't exist. Use the menu to pick a tool.</div>} />
        </Routes>
      </main>
    </div>
  );
}

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <BrowserRouter>
      <Shell />
    </BrowserRouter>
  </StrictMode>,
);
