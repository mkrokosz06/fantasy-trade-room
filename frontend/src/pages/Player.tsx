import { useParams } from "react-router-dom";
import { Bar, CartesianGrid, ComposedChart, Legend, Line, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { fmt, pct, useApi, type Call, type Meta, type MlStatus } from "../api";
import { CallBadge, ErrorBox, InjuryBadge, Loading, Range } from "../components/bits";
import { useEngine } from "../engine";

const GROUP: Record<string, string> = { QB: "QB", RB: "RB", WR: "REC", TE: "REC" };

interface Detail {
  info: { id: string; name: string; position: string; team: string | null; age: number | null; injury_status: string | null; injury_body_part: string | null; headshot: string | null };
  value?: { value: number; vorp: number; ros_pts: number; proj_ppg: number; pos_rank: number; overall_rank: number; replacement: number; games_left: number; floor_ppg?: number; ceiling_ppg?: number };
  game_log: Record<string, number | string | null>[];
  seasons: Record<string, number | string | null>[];
  projection: { week: number; opp: string; is_home: boolean; gameday: string; matchup: number; vegas: number; avail: number; sleeper_weight: number; pts: number; floor?: number; ceiling?: number; pass_yd: number; pass_td: number; rush_att: number; rush_yd: number; rush_td: number; rec_tgt: number; rec: number; rec_yd: number; rec_td: number }[];
  model?: { weighted_games: number; per_game: Record<string, number>; efficiency: { stat: string; per: string; player: number; position_avg: number | null }[] };
  signal?: { call: Call; reasons: string[]; edge: number; confidence: number };
}

const EFF_LABEL: Record<string, string> = {
  pass_cmp: "Completion rate", pass_yd: "Yards per attempt", pass_td: "TD per attempt", pass_int: "INT per attempt",
  rush_yd: "Yards per carry", rush_td: "TD per carry", rec: "Catch rate", rec_yd: "Yards per target", rec_td: "TD per target",
};

export default function Player() {
  const { id = "" } = useParams();
  const { ep, isModel } = useEngine();
  const meta = useApi<Meta>("/api/meta");
  const d = useApi<Detail>(ep(`/api/players/${encodeURIComponent(id)}`));
  const ml = useApi<MlStatus>(isModel ? "/api/ml/status" : null);
  if (d.error) return <ErrorBox error={d.error} />;
  if (!d.data) return <Loading what="Loading player" />;
  const { info, value, signal, model } = d.data;
  const season = meta.data?.season;
  const thisSeason = d.data.game_log.filter((g) => g.season === season).slice().reverse();
  const pos = info.position;
  const relevantEff = (model?.efficiency ?? []).filter((e) =>
    pos === "QB" ? e.per !== "rec_tgt" : pos === "RB" ? e.per !== "pass_att" : e.per === "rec_tgt",
  );
  // SVG attributes can't resolve CSS variables, so read the theme colors directly.
  const css = getComputedStyle(document.documentElement);
  const color = (v: string) => css.getPropertyValue(v).trim();

  return (
    <div className="stack">
      <section className="player-hero">
        <img src={info.headshot || undefined} alt="" onError={(e) => ((e.target as HTMLImageElement).style.visibility = "hidden")} />
        <div className="grow">
          <h1>{info.name}</h1>
          <div className="row" style={{ marginTop: 4 }}>
            <span className="pos">
              {info.position} {info.team ?? "Free agent"}
            </span>
            {info.age && <span className="muted small">Age {Math.floor(info.age)}</span>}
            <InjuryBadge status={info.injury_status} />
            {info.injury_body_part && <span className="small muted">{info.injury_body_part}</span>}
            <CallBadge call={signal?.call} />
          </div>
        </div>
        {value && (
          <div className="kpis">
            <div className="stat"><b>{fmt(value.value, 0)}</b><span>Value</span></div>
            <div className="stat"><b>{fmt(value.ros_pts, 0)}</b><span>ROS points</span></div>
            <div className="stat"><b>{fmt(value.proj_ppg)}</b><span>Proj PPG</span></div>
            {value.floor_ppg !== undefined && (
              <div className="stat" title="In a typical week, 80% of games land in this range">
                <b><Range floor={value.floor_ppg} ceiling={value.ceiling_ppg} /></b><span>Weekly range</span>
              </div>
            )}
            <div className="stat"><b>{pos}{value.pos_rank}</b><span>Position rank</span></div>
          </div>
        )}
      </section>

      {signal && (
        <section className="panel">
          <h3>Why {signal.call === "BUY LOW" ? "buy low" : signal.call === "SELL HIGH" ? "sell high" : "hold"}</h3>
          <ul>{signal.reasons.map((r, i) => <li key={i}>{r}</li>)}</ul>
        </section>
      )}

      {thisSeason.length > 0 && (
        <section className="panel">
          <h2>{season} points vs expected</h2>
          <p className="small muted">Bars are actual points. The line is expected points from this player's usage at league-average efficiency.</p>
          <div className="chart">
            <ResponsiveContainer>
              <ComposedChart data={thisSeason} margin={{ top: 12, right: 8, left: -18, bottom: 0 }}>
                <CartesianGrid stroke={color("--line")} vertical={false} />
                <XAxis dataKey="week" tickFormatter={(w) => `W${w}`} stroke={color("--muted")} fontSize={12} />
                <YAxis stroke={color("--muted")} fontSize={12} />
                <Tooltip
                  contentStyle={{ background: color("--surface"), border: `1px solid ${color("--line-strong")}`, borderRadius: 8 }}
                  labelFormatter={(w) => `Week ${w}`}
                  formatter={(v) => (typeof v === "number" ? v.toFixed(1) : v)}
                />
                <Legend />
                <Bar dataKey="pts" name="Actual" fill={color("--turf-soft")} radius={[3, 3, 0, 0]} isAnimationActive={false} />
                <Line dataKey="xfp" name="Expected" stroke={color("--flag")} strokeWidth={2.5} dot={{ r: 3, fill: color("--flag") }} isAnimationActive={false} />
              </ComposedChart>
            </ResponsiveContainer>
          </div>
        </section>
      )}

      <div className="grid2">
        <section className="panel">
          <h2>Upcoming projections</h2>
          {d.data.projection.length ? (
            <div className="tablewrap" style={{ marginTop: 10 }}>
              <table>
                <thead>
                  <tr>
                    <th>Wk</th><th>Opp</th><th className="n" title="Opponent factor: above 1 means the defense allows more points to this position">Matchup</th>
                    <th className="n" title="Chance of playing, from injury status">Plays</th><th className="n">Pts</th>
                    {isModel && <th className="n" title="80% of the time the result lands in this range">Range</th>}
                  </tr>
                </thead>
                <tbody>
                  {d.data.projection.map((p) => (
                    <tr key={p.week}>
                      <td>{p.week}</td>
                      <td>{p.is_home ? "vs" : "@"} {p.opp}</td>
                      <td className="n" style={{ color: p.matchup > 1.04 ? "var(--good)" : p.matchup < 0.96 ? "var(--bad)" : undefined }}>{fmt(p.matchup, 2)}</td>
                      <td className="n">{pct(p.avail)}</td>
                      <td className="n"><b>{fmt(p.pts)}</b></td>
                      {isModel && <td className="n muted"><Range floor={p.floor} ceiling={p.ceiling} /></td>}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <p className="muted">No remaining games projected.</p>
          )}
        </section>

        {isModel && ml.data?.meta && GROUP[pos] && (
          <section className="panel">
            <h2>What drives the model for {pos === "WR" || pos === "TE" ? "pass catchers" : `${pos}s`}</h2>
            <p className="small muted">Share of the model's predictive power, from the PPR points model.</p>
            <div className="bars">
              {ml.data.meta.importance[GROUP[pos]].slice(0, 8).map((f) => (
                <div key={f.feature} className="bar-row">
                  <span>{f.label}</span>
                  <i style={{ width: `${Math.min(100, (f.share / ml.data!.meta!.importance[GROUP[pos]][0].share) * 100)}%` }} />
                  <span className="num small muted">{(f.share * 100).toFixed(0)}%</span>
                </div>
              ))}
            </div>
          </section>
        )}

        {!isModel && model && pos !== "K" && (
          <section className="panel">
            <h2>How the projection is built</h2>
            <p className="small muted">
              Recent-weighted usage (about {fmt(model.weighted_games, 1)} games of evidence), with efficiency pulled toward the {pos} average
              until the sample is large enough to trust.
            </p>
            <div className="row" style={{ gap: 24, margin: "10px 0" }}>
              {pos === "QB" && <div className="stat"><b>{fmt(model.per_game.pass_att)}</b><span>Pass att/game</span></div>}
              <div className="stat"><b>{fmt(model.per_game.rush_att)}</b><span>Carries/game</span></div>
              {pos !== "QB" && <div className="stat"><b>{fmt(model.per_game.rec_tgt)}</b><span>Targets/game</span></div>}
            </div>
            <div className="tablewrap">
              <table>
                <thead><tr><th>Efficiency</th><th className="n">Projected</th><th className="n">{pos} avg</th></tr></thead>
                <tbody>
                  {relevantEff.map((e) => (
                    <tr key={e.stat}>
                      <td>{EFF_LABEL[e.stat] ?? e.stat}</td>
                      <td className="n">{e.stat.endsWith("_yd") ? fmt(e.player, 2) : pct(e.player, 1)}</td>
                      <td className="n muted">{e.position_avg === null ? "–" : e.stat.endsWith("_yd") ? fmt(e.position_avg, 2) : pct(e.position_avg, 1)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>
        )}
      </div>

      <section className="stack">
        <h2>Game log</h2>
        <div className="tablewrap">
          <table>
            <thead>
              <tr>
                <th>Season</th><th>Wk</th><th>Opp</th><th className="n">Pts</th><th className="n">xFP</th>
                {pos === "QB" && <><th className="n">Cmp/Att</th><th className="n">Pass yds</th><th className="n">Pass TD</th><th className="n">INT</th></>}
                {pos !== "K" && <><th className="n">Car</th><th className="n">Rush yds</th><th className="n">Rush TD</th></>}
                {pos !== "QB" && pos !== "K" && <><th className="n">Tgt</th><th className="n">Rec</th><th className="n">Rec yds</th><th className="n">Rec TD</th><th className="n">Tgt share</th></>}
                {pos === "K" && <><th className="n">FG 0-39</th><th className="n">FG 40-49</th><th className="n">FG 50+</th><th className="n">Missed</th><th className="n">XP</th></>}
              </tr>
            </thead>
            <tbody>
              {d.data.game_log.map((g) => (
                <tr key={`${g.season}-${g.week}`}>
                  <td>{g.season}</td><td>{g.week}</td><td>{g.opp}</td>
                  <td className="n"><b>{fmt(g.pts as number)}</b></td><td className="n muted">{fmt(g.xfp as number)}</td>
                  {pos === "QB" && <><td className="n">{g.pass_cmp}/{g.pass_att}</td><td className="n">{g.pass_yd}</td><td className="n">{g.pass_td}</td><td className="n">{g.pass_int}</td></>}
                  {pos !== "K" && <><td className="n">{g.rush_att}</td><td className="n">{g.rush_yd}</td><td className="n">{g.rush_td}</td></>}
                  {pos !== "QB" && pos !== "K" && <><td className="n">{g.rec_tgt}</td><td className="n">{g.rec}</td><td className="n">{g.rec_yd}</td><td className="n">{g.rec_td}</td><td className="n">{pct(g.target_share as number)}</td></>}
                  {pos === "K" && <><td className="n">{g.fg_0_39}</td><td className="n">{g.fg_40_49}</td><td className="n">{g.fg_50p}</td><td className="n">{g.fg_miss}</td><td className="n">{g.xp_made}</td></>}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  );
}
