import { useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { fmt, pct, useApi } from "../api";
import { useEngine } from "../engine";
import { ErrorBox } from "./bits";

type Game = Record<string, number | string | null>;
interface Detail {
  info: { position: string };
  game_log: Game[];
}

const n = (v: number | string | null | undefined) => (typeof v === "number" ? v : 0);

/** Compact game log with fantasy points (scored with the user's league settings). */
export function GameLog({ id, lastWeek }: { id: string; lastWeek?: number }) {
  const { ep, base } = useEngine();
  const d = useApi<Detail>(ep(`/api/players/${encodeURIComponent(id)}`));
  const seasons = useMemo(
    () => [...new Set((d.data?.game_log ?? []).map((g) => g.season as number))].sort((a, b) => b - a),
    [d.data],
  );
  const [season, setSeason] = useState<number | null>(null);
  const s = season ?? seasons[0];

  if (d.error) return <ErrorBox error={d.error} />;
  if (!d.data) return <div className="gamelog muted small">Loading game log…</div>;
  if (!seasons.length) return <div className="gamelog muted small">No NFL games on record yet.</div>;

  const pos = d.data.info.position;
  const games = d.data.game_log.filter((g) => g.season === s).sort((a, b) => n(a.week) - n(b.week));
  // Weeks with no game: bye, injury or inactive. Only filled in for the current season.
  const isCurrent = s === seasons[0] && lastWeek;
  const played = new Set(games.map((g) => g.week));
  const rows: (Game | { week: number; missed: true })[] = [];
  const maxWeek = isCurrent ? lastWeek! : Math.max(...games.map((g) => n(g.week)));
  for (let w = 1; w <= maxWeek; w++) {
    const g = games.find((x) => x.week === w);
    if (g) rows.push(g);
    else if (isCurrent && !played.has(w)) rows.push({ week: w, missed: true });
  }
  rows.reverse(); // most recent first

  // Flag games the player left early (well under their usual volume), so a
  // zero from an injury exit doesn't read like a bad game.
  const opps = (g: Game) => n(g.pass_att) + n(g.rush_att) + n(g.rec_tgt);
  const vols = games.map(opps).sort((a, b) => a - b);
  const typical = vols.length ? vols[Math.floor(vols.length / 2)] : 0;
  const leftEarly = (g: Game) => pos !== "K" && typical >= 6 && opps(g) < 0.35 * typical;

  const pts = games.map((g) => n(g.pts));
  const ppg = pts.length ? pts.reduce((a, b) => a + b, 0) / pts.length : 0;
  const last3 = pts.slice(-3);
  const l3 = last3.length ? last3.reduce((a, b) => a + b, 0) / last3.length : 0;
  const xfp = games.map((g) => n(g.xfp));
  const xppg = xfp.length ? xfp.reduce((a, b) => a + b, 0) / xfp.length : 0;

  const cols: { label: string; title?: string; v: (g: Game) => string | number }[] =
    pos === "QB"
      ? [
          { label: "Cmp/Att", v: (g) => `${n(g.pass_cmp)}/${n(g.pass_att)}` },
          { label: "Pass yds", v: (g) => n(g.pass_yd) },
          { label: "TD", v: (g) => n(g.pass_td) },
          { label: "INT", v: (g) => n(g.pass_int) },
          { label: "Car", v: (g) => n(g.rush_att) },
          { label: "Rush yds", v: (g) => n(g.rush_yd) },
          { label: "Rush TD", v: (g) => n(g.rush_td) },
        ]
      : pos === "RB"
        ? [
            { label: "Car", v: (g) => n(g.rush_att) },
            { label: "Rush yds", v: (g) => n(g.rush_yd) },
            { label: "Rush TD", v: (g) => n(g.rush_td) },
            { label: "Tgt", v: (g) => n(g.rec_tgt) },
            { label: "Rec", v: (g) => n(g.rec) },
            { label: "Rec yds", v: (g) => n(g.rec_yd) },
            { label: "Rec TD", v: (g) => n(g.rec_td) },
          ]
        : pos === "K"
          ? [
              { label: "FG 0-39", v: (g) => n(g.fg_0_39) },
              { label: "FG 40-49", v: (g) => n(g.fg_40_49) },
              { label: "FG 50+", v: (g) => n(g.fg_50p) },
              { label: "Missed", v: (g) => n(g.fg_miss) },
              { label: "XP", v: (g) => n(g.xp_made) },
            ]
          : [
              { label: "Tgt", v: (g) => n(g.rec_tgt) },
              { label: "Rec", v: (g) => n(g.rec) },
              { label: "Rec yds", v: (g) => n(g.rec_yd) },
              { label: "TD", v: (g) => n(g.rec_td) },
              { label: "Tgt share", title: "Share of the team's targets", v: (g) => pct(g.target_share as number) },
              { label: "Car", v: (g) => n(g.rush_att) },
              { label: "Rush yds", v: (g) => n(g.rush_yd) },
            ];

  return (
    <div className="gamelog">
      <div className="row" style={{ justifyContent: "space-between" }}>
        <div className="row" style={{ gap: 18 }}>
          <span className="small"><b>{fmt(ppg)}</b> <span className="muted">PPG</span></span>
          <span className="small"><b>{fmt(l3)}</b> <span className="muted">last 3</span></span>
          {pos !== "K" && (
            <span className="small" title="Expected points per game from usage at league-average efficiency">
              <b>{fmt(xppg)}</b> <span className="muted">expected PPG</span>
            </span>
          )}
          <span className="small"><b>{games.length}</b> <span className="muted">games</span></span>
        </div>
        <div className="row" style={{ gap: 8 }}>
          {seasons.length > 1 && (
            <select className="input small" value={s} onChange={(e) => setSeason(Number(e.target.value))} aria-label="Season">
              {seasons.map((x) => <option key={x} value={x}>{x}</option>)}
            </select>
          )}
          <Link className="small" to={`${base}/player/${encodeURIComponent(id)}`}>Full player page</Link>
        </div>
      </div>
      <div className="tablewrap" style={{ marginTop: 8 }}>
        <table>
          <thead>
            <tr>
              <th>Wk</th>
              <th>Opp</th>
              <th className="n" title="Fantasy points with your league's scoring">Pts</th>
              {pos !== "K" && <th className="n" title="Expected points from this game's usage">xFP</th>}
              {cols.map((c) => <th key={c.label} className="n" title={c.title}>{c.label}</th>)}
            </tr>
          </thead>
          <tbody>
            {rows.map((g) =>
              "missed" in g ? (
                <tr key={`m${g.week}`} className="muted">
                  <td>{g.week}</td>
                  <td colSpan={cols.length + (pos !== "K" ? 3 : 2)} className="small">No game: bye week, injured or inactive</td>
                </tr>
              ) : (
                <tr key={String(g.week)}>
                  <td>{g.week}</td>
                  <td>
                    {g.opp}
                    {leftEarly(g) && (
                      <span className="badge inj" style={{ marginLeft: 6 }} title="Far below usual playing time, likely an injury exit">
                        left early
                      </span>
                    )}
                  </td>
                  <td className="n"><b className={n(g.pts) >= ppg * 1.25 ? "delta up" : n(g.pts) <= ppg * 0.6 ? "delta down" : ""}>{fmt(n(g.pts))}</b></td>
                  {pos !== "K" && <td className="n muted">{fmt(n(g.xfp))}</td>}
                  {cols.map((c) => <td key={c.label} className="n">{c.v(g)}</td>)}
                </tr>
              ),
            )}
          </tbody>
        </table>
      </div>
      <p className="small muted" style={{ marginTop: 6 }}>
        Points use your league's scoring. Green games are well above this player's average, red well below.
      </p>
    </div>
  );
}
