import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api, fmt, invalidateAll, odds, useApi, type League, type LeagueOdds, type Meta, type Settings } from "../api";
import { CallBadge, ErrorBox, Loading } from "../components/bits";
import { SituationBadge } from "../components/OddsBits";

interface HistoryTrade {
  period: number;
  sides: { team_id: string; team_name: string; pts_since: number; ros_value: number;
    received: { id: string; name: string; pts_since: number; ros_value: number }[] }[];
}

interface OfferRow {
  period: number;
  status: string;
  proposer: string;
  proposer_name: string;
  sides: { team_id: string; team_name: string; receives: { id: string; name: string }[] }[];
  drops: { id: string; name: string }[];
}

function Offers({ offers, me }: { offers: OfferRow[]; me: string | null }) {
  if (!offers.length) return null;
  const statusCls = (s: string) => (s === "Accepted" ? "buy" : s === "Pending" ? "hold" : "sell");
  return (
    <div className="stack" style={{ gap: 8 }}>
      <h3>Trade offers this season</h3>
      <div className="tablewrap">
        <table>
          <thead>
            <tr><th>Wk</th><th>From</th><th>Deal</th><th>Status</th><th></th></tr>
          </thead>
          <tbody>
            {offers.map((o, i) => {
              const other = o.sides.find((s) => s.team_id !== o.proposer);
              const mine = o.sides.find((s) => s.team_id === o.proposer);
              const give = other?.receives.map((p) => p.id) ?? [];
              const get = mine?.receives.map((p) => p.id) ?? [];
              const involvesMe = me !== null && o.sides.some((s) => s.team_id === me);
              return (
                <tr key={i} style={involvesMe ? { background: "var(--buy-bg)" } : undefined}>
                  <td>{o.period}</td>
                  <td><b>{o.proposer_name}</b>{involvesMe && <span className="badge hold" style={{ marginLeft: 6 }}>Involves you</span>}</td>
                  <td style={{ whiteSpace: "normal" }}>
                    {o.sides.map((s) => (
                      <div key={s.team_id} className="small">
                        <b>{s.team_name}</b> gets {s.receives.map((p) => p.name).join(", ")}
                      </div>
                    ))}
                    {o.drops.length > 0 && (
                      <div className="small muted">Dropped to make room: {o.drops.map((p) => p.name).join(", ")}</div>
                    )}
                  </td>
                  <td><span className={`badge ${statusCls(o.status)}`}>{o.status}</span></td>
                  <td>
                    {give.length > 0 && get.length > 0 && other && (
                      <Link className="btn ghost small" to={`/?a=${give.join(",")}&b=${get.join(",")}&ta=${o.proposer}&tb=${other.team_id}`}>
                        Open in calculator
                      </Link>
                    )}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function Meter({ p }: { p: number }) {
  return <span className="meter" aria-hidden="true"><i style={{ width: `${Math.round(p * 100)}%` }} /></span>;
}

function Standings({ lg }: { lg: League }) {
  const o = useApi<LeagueOdds>("/api/league/odds");
  const h = useApi<{ trades: HistoryTrade[]; offers: OfferRow[] }>("/api/league/history");
  const lineup = new Map(lg.teams.map((t) => [t.team_id, t.lineup_ros_pts]));
  if (o.error) return <ErrorBox error={o.error} />;
  if (!o.data) return <Loading what="Simulating the rest of the season" />;
  const r = o.data.league_accept_rate;
  return (
    <section className="stack">
      <div>
        <h2>Standings and odds</h2>
        <p className="small muted">
          From {o.data.sims.toLocaleString()} simulations of the rest of the season: each team's best weekly lineup
          ({o.data.engine === "ml" ? "trained model" : "original engine"} projections with week-to-week swings), the real
          schedule, then a {o.data.playoffs.teams}-team playoff after week {o.data.playoffs.regular_season_weeks}.
        </p>
      </div>
      <div className="tablewrap">
        <table>
          <thead>
            <tr>
              <th>Team</th>
              <th className="n">Record</th>
              <th className="n">Points for</th>
              <th className="n" title="Average final win total across simulations">Proj. wins</th>
              <th title="Chance of making the playoffs">Playoffs</th>
              <th className="n" title="Chance of a first-round bye">Bye</th>
              <th className="n" title="Chance of winning the league">Title</th>
              <th>Situation</th>
              <th className="n" title="Trade offers sent / trades completed this season">Trading</th>
              <th className="n" title="Projected best starting lineup, rest of season">Lineup ROS pts</th>
            </tr>
          </thead>
          <tbody>
            {o.data.standings.map((t) => (
              <tr key={t.team_id}>
                <td>
                  <b>{t.name}</b>
                  {t.team_id === lg.my_team_id && <span className="badge hold" style={{ marginLeft: 8 }}>You</span>}
                  <div className="small muted">{t.owner}</div>
                </td>
                <td className="n">{t.wins}-{t.losses}{t.ties ? `-${t.ties}` : ""}</td>
                <td className="n">{fmt(t.points_for)}</td>
                <td className="n">{t.avg_wins.toFixed(1)}</td>
                <td className="nowrap"><Meter p={t.playoff} /><b>{odds(t.playoff)}</b></td>
                <td className="n">{odds(t.bye)}</td>
                <td className="n"><b>{odds(t.title)}</b></td>
                <td><SituationBadge s={t.situation} /></td>
                <td className="n">{t.activity.proposed} / {t.activity.completed}</td>
                <td className="n">{fmt(lineup.get(t.team_id), 0)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <section className="panel stack">
        <h3>Trade history</h3>
        <p className="small muted">
          {r.proposals} trade offers this season, {r.completed} accepted.
          {r.completed === 0 && " Offers in this league are rarely accepted, so the advisor's acceptance estimates are scaled down to match."}
        </p>
        {h.data && <Offers offers={h.data.offers ?? []} me={lg.my_team_id} />}
        {h.data?.trades.map((t, i) => (
          <div key={i} className="odds-card">
            <b>Week {t.period}</b>
            <div className="grid2" style={{ marginTop: 6 }}>
              {t.sides.map((s) => (
                <div key={s.team_id}>
                  <div><b>{s.team_name}</b> received</div>
                  <ul className="small" style={{ margin: "4px 0", paddingLeft: 18 }}>
                    {s.received.map((p) => <li key={p.id}>{p.name}: {fmt(p.pts_since)} pts since</li>)}
                  </ul>
                  <div className="small muted">{fmt(s.pts_since)} pts since the trade, {fmt(s.ros_value, 0)} value rest of season</div>
                </div>
              ))}
            </div>
          </div>
        ))}
      </section>
    </section>
  );
}

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

          <Standings lg={lg} />

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
