import { useEffect, useState } from "react";
import { api, invalidateAll, useApi, type Meta, type Settings } from "../api";
import { ErrorBox, Loading } from "../components/bits";

interface Presets {
  scoring: Record<string, { label: string; rules: Record<string, number>; te_rec_bonus: number }>;
  league: Record<string, { label: string; teams: number; slots: Record<string, number> }>;
  slot_labels: Record<string, string>;
  stat_labels: Record<string, string>;
}

const SLOT_ORDER = ["QB", "RB", "WR", "TE", "FLEX", "WRRB", "WRTE", "SFLEX", "K", "DST", "BENCH"];

export default function SettingsPage() {
  const saved = useApi<Settings>("/api/settings");
  const presets = useApi<Presets>("/api/presets");
  const meta = useApi<Meta>("/api/meta");
  const [s, setS] = useState<Settings | null>(null);
  const [status, setStatus] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    if (saved.data) setS(structuredClone(saved.data));
  }, [saved.data]);

  if (!s || !presets.data) return <Loading what="Loading settings" />;
  const P = presets.data;
  const dirty = JSON.stringify(s) !== JSON.stringify(saved.data);

  const save = async () => {
    setBusy(true);
    setErr(null);
    try {
      await api("/api/settings", { method: "PUT", body: JSON.stringify(s) });
      invalidateAll();
      setStatus("Settings saved. Values and projections now use them.");
    } catch (e) {
      setErr((e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  const refresh = async () => {
    setBusy(true);
    setErr(null);
    setStatus("Downloading fresh data…");
    try {
      await api("/api/refresh", { method: "POST" });
      invalidateAll();
      setStatus("Data refreshed.");
    } catch (e) {
      setErr((e as Error).message);
      setStatus(null);
    } finally {
      setBusy(false);
    }
  };

  const num = (v: string) => (v === "" || Number.isNaN(Number(v)) ? 0 : Number(v));

  return (
    <div className="stack">
      <div className="page-head">
        <div>
          <h1>Settings</h1>
          <p>Scoring and lineup slots change every value, projection and recommendation in the app.</p>
        </div>
        <div className="row">
          <button className="btn ghost" onClick={() => setS(structuredClone(saved.data!))} disabled={!dirty || busy}>Discard changes</button>
          <button className="btn" onClick={save} disabled={!dirty || busy}>Save settings</button>
        </div>
      </div>
      <ErrorBox error={err || saved.error} />
      {status && <div className="panel small">{status}</div>}

      <div className="grid2">
        <section className="panel stack">
          <h2>Scoring</h2>
          <label className="row">
            <span className="grow">Preset</span>
            <select
              className="input"
              value={P.scoring[s.scoring.preset] ? s.scoring.preset : ""}
              onChange={(e) => {
                const p = P.scoring[e.target.value];
                if (p) setS({ ...s, scoring: { preset: e.target.value, ...structuredClone(p) } });
              }}
            >
              {!P.scoring[s.scoring.preset] && <option value="">{s.scoring.label || "Custom"}</option>}
              {Object.entries(P.scoring).map(([k, v]) => (
                <option key={k} value={k}>{v.label}</option>
              ))}
            </select>
          </label>
          <div className="tablewrap">
            <table>
              <tbody>
                {Object.entries(s.scoring.rules).map(([k, v]) => (
                  <tr key={k}>
                    <td>{P.stat_labels[k] ?? k}</td>
                    <td className="n">
                      <input
                        className="input num"
                        style={{ width: 90 }}
                        type="number"
                        step="0.01"
                        value={v}
                        aria-label={`Points for ${P.stat_labels[k] ?? k}`}
                        onChange={(e) =>
                          setS({ ...s, scoring: { ...s.scoring, preset: "custom", label: "Custom", rules: { ...s.scoring.rules, [k]: num(e.target.value) } } })
                        }
                      />
                    </td>
                  </tr>
                ))}
                <tr>
                  <td>TE bonus per reception</td>
                  <td className="n">
                    <input
                      className="input num"
                      style={{ width: 90 }}
                      type="number"
                      step="0.25"
                      value={s.scoring.te_rec_bonus}
                      aria-label="TE bonus per reception"
                      onChange={(e) => setS({ ...s, scoring: { ...s.scoring, preset: "custom", label: "Custom", te_rec_bonus: num(e.target.value) } })}
                    />
                  </td>
                </tr>
              </tbody>
            </table>
          </div>
          <p className="small muted">Yardage is points per yard: 0.04 means 1 point per 25 passing yards.</p>
        </section>

        <div className="stack">
          <section className="panel stack">
            <h2>League format</h2>
            <label className="row">
              <span className="grow">Preset</span>
              <select
                className="input"
                value={P.league[s.league.preset] ? s.league.preset : ""}
                onChange={(e) => {
                  const p = P.league[e.target.value];
                  if (p) setS({ ...s, league: { preset: e.target.value, ...structuredClone(p) } });
                }}
              >
                {!P.league[s.league.preset] && <option value="">{s.league.label || "Custom"}</option>}
                {Object.entries(P.league).map(([k, v]) => (
                  <option key={k} value={k}>{v.label}</option>
                ))}
              </select>
            </label>
            <label className="row">
              <span className="grow">Teams</span>
              <input className="input num" style={{ width: 90 }} type="number" min={2} max={32} value={s.league.teams}
                onChange={(e) => setS({ ...s, league: { ...s.league, preset: "custom", label: "Custom", teams: num(e.target.value) } })} />
            </label>
            <div className="tablewrap">
              <table>
                <tbody>
                  {SLOT_ORDER.map((k) => (
                    <tr key={k}>
                      <td>{P.slot_labels[k] ?? k}</td>
                      <td className="n">
                        <input
                          className="input num"
                          style={{ width: 90 }}
                          type="number"
                          min={0}
                          value={s.league.slots[k] ?? 0}
                          aria-label={`${P.slot_labels[k] ?? k} slots`}
                          onChange={(e) =>
                            setS({ ...s, league: { ...s.league, preset: "custom", label: "Custom", slots: { ...s.league.slots, [k]: num(e.target.value) } } })
                          }
                        />
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>

          <section className="panel stack">
            <h2>Projections</h2>
            <label className="stack" style={{ gap: 4 }}>
              <span>Sleeper blend: {Math.round(s.projection.sleeper_blend * 100)}% Sleeper, {Math.round((1 - s.projection.sleeper_blend) * 100)}% this app's model</span>
              <input type="range" min={0} max={1} step={0.05} value={s.projection.sleeper_blend}
                onChange={(e) => setS({ ...s, projection: { ...s.projection, sleeper_blend: Number(e.target.value) } })} />
              <span className="small muted">Players with little history (rookies) lean on Sleeper automatically.</span>
            </label>
            <label className="row">
              <span className="grow">Count rest of season through week</span>
              <input className="input num" style={{ width: 90 }} type="number" min={1} max={18} value={s.projection.end_week}
                onChange={(e) => setS({ ...s, projection: { ...s.projection, end_week: num(e.target.value) } })} />
            </label>
            <label className="row">
              <span className="grow">Recency half-life (games)</span>
              <input className="input num" style={{ width: 90 }} type="number" min={1} max={30} step={0.5} value={s.projection.half_life_games}
                onChange={(e) => setS({ ...s, projection: { ...s.projection, half_life_games: num(e.target.value) } })} />
            </label>
            <span className="small muted">Lower half-life reacts faster to role changes but is noisier.</span>
          </section>

          <section className="panel stack">
            <h2>Data</h2>
            <p className="small muted">
              {meta.data
                ? `Season ${meta.data.season}, stats through week ${meta.data.last_stats_week}. Loaded ${new Date(meta.data.loaded_at).toLocaleString()}.`
                : "Loading…"}{" "}
              Data refreshes automatically every 12 hours. nflverse updates overnight after games.
            </p>
            {meta.data?.warnings.map((w) => <p key={w} className="small muted">{w}</p>)}
            <div><button className="btn ghost" onClick={refresh} disabled={busy}>Refresh data now</button></div>
          </section>
        </div>
      </div>
    </div>
  );
}
