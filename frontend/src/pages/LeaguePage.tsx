import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api, fmt, invalidateAll, useApi, type League, type Meta, type Settings } from "../api";
import { CallBadge, ErrorBox, Loading } from "../components/bits";

export default function LeaguePage() {
  const settings = useApi<Settings>("/api/settings");
  const meta = useApi<Meta>("/api/meta");
  const leagueId = settings.data?.espn.league_id;
  const league = useApi<League>(leagueId ? "/api/league" : null);
  const [idInput, setIdInput] = useState("");
  const [msg, setMsg] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    if (leagueId) setIdInput(leagueId);
  }, [leagueId]);

  const saveEspn = async (patch: Partial<Settings["espn"]>) => {
    if (!settings.data) return;
    const next = { ...settings.data, espn: { ...settings.data.espn, ...patch } };
    await api("/api/settings", { method: "PUT", body: JSON.stringify(next) });
    invalidateAll();
  };

  const connect = async () => {
    setErr(null);
    setMsg(null);
    const id = idInput.trim();
    if (!/^\d+$/.test(id)) {
      setErr("League ID should be the number after leagueId= in your ESPN league URL.");
      return;
    }
    await saveEspn({ league_id: id, my_team_id: "" });
  };

  const importSettings = async () => {
    setBusy(true);
    setErr(null);
    try {
      const r = await api<{ imported: League["settings"] & { league: { teams: number; slots: Record<string, number> } } }>(
        "/api/league/import-settings",
        { method: "POST" },
      );
      const um = r.imported.unmapped;
      setMsg(
        `Imported scoring and lineup slots for ${r.imported.league.teams} teams.` +
          (um.stats.length ? ` ${um.stats.length} scoring rules were not recognized and are not applied (stat ids ${um.stats.map((s) => s.stat_id).join(", ")}).` : ""),
      );
      invalidateAll();
    } catch (e) {
      setErr((e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  const lg = league.data;
  return (
    <div className="stack">
      <div className="page-head">
        <div>
          <h1>ESPN league</h1>
          <p>Connect your league to see every roster, use real teams in the trade calculator, and get trade ideas.</p>
        </div>
      </div>

      <section className="panel stack">
        <div className="row">
          <label htmlFor="lid"><b>League ID</b></label>
          <input id="lid" className="input" value={idInput} onChange={(e) => setIdInput(e.target.value)} placeholder="e.g. 12345678" />
          <button className="btn" onClick={connect}>{leagueId ? "Update league" : "Connect league"}</button>
          {lg && <button className="btn ghost" onClick={() => league.reload()}>Reload rosters</button>}
        </div>
        <p className="small muted">
          Find it in your ESPN league URL: fantasy.espn.com/football/league?leagueId=<b>12345678</b>. Private leagues
          also need two login cookies in the <code>.env</code> file (see the README).{" "}
          {meta.data && (meta.data.espn_cookies_set ? "Cookies are set." : "Cookies are not set, so only public leagues will load.")}
        </p>
        <ErrorBox error={err || league.error} />
        {msg && <div className="panel small">{msg}</div>}
      </section>

      {leagueId && !lg && !league.error && <Loading what="Loading league from ESPN" />}

      {lg && (
        <>
          <section className="panel stack">
            <div className="side-head">
              <h2>{lg.name}</h2>
              <span className="muted">{lg.size} teams</span>
            </div>
            <div className="row">
              <label htmlFor="myteam"><b>Your team</b></label>
              <select
                id="myteam"
                className="input"
                value={lg.my_team_id ?? ""}
                onChange={(e) => saveEspn({ my_team_id: e.target.value })}
              >
                <option value="">Choose your team</option>
                {lg.teams.map((t) => (
                  <option key={t.team_id} value={t.team_id}>{t.name} ({t.owner})</option>
                ))}
              </select>
              <button className="btn ghost" onClick={importSettings} disabled={busy}>
                {busy ? "Importing…" : "Use this league's scoring and lineup"}
              </button>
            </div>
            {lg.settings.unmapped.stats.length > 0 && (
              <p className="small muted">
                {lg.settings.unmapped.stats.length} of this league's scoring rules aren't modeled yet (for example
                yardage bonuses or D/ST stats), so they won't count in projections.
              </p>
            )}
          </section>

          <div className="tablewrap">
            <table>
              <thead>
                <tr>
                  <th>Team</th><th>Owner</th><th className="n">Record</th><th className="n">Points for</th>
                  <th className="n" title="Projected best starting lineup, rest of season">Lineup ROS pts</th>
                </tr>
              </thead>
              <tbody>
                {[...lg.teams].sort((a, b) => b.lineup_ros_pts - a.lineup_ros_pts).map((t) => (
                  <tr key={t.team_id}>
                    <td><b>{t.name}</b>{t.team_id === lg.my_team_id && <span className="badge hold" style={{ marginLeft: 8 }}>You</span>}</td>
                    <td>{t.owner}</td>
                    <td className="n">{t.wins}-{t.losses}{t.ties ? `-${t.ties}` : ""}</td>
                    <td className="n">{fmt(t.points_for)}</td>
                    <td className="n"><b>{fmt(t.lineup_ros_pts, 0)}</b></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div className="grid2">
            {lg.teams.map((t) => (
              <details key={t.team_id} className="panel" open={t.team_id === lg.my_team_id}>
                <summary><b>{t.name}</b> <span className="muted small">{t.owner}</span></summary>
                <table style={{ marginTop: 8 }}>
                  <tbody>
                    {t.roster.map((p) => (
                      <tr key={p.espn_id}>
                        <td className="pos">{p.slot}</td>
                        <td>
                          {p.id ? <Link to={`/player/${encodeURIComponent(p.id)}`}>{p.name}</Link> : p.name}{" "}
                          <span className="pos">{p.position} {p.pro_team}</span>
                        </td>
                        <td><CallBadge call={p.call} /></td>
                        <td className="n">{p.value !== undefined ? fmt(p.value, 0) : "–"}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </details>
            ))}
          </div>
        </>
      )}
    </div>
  );
}
