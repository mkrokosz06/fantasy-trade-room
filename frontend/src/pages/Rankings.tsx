import { useState } from "react";
import { fmt, pct, useApi, type Meta, type RankedPlayer } from "../api";
import { ErrorBox, Loading, PlayerCell, PosFilter, Range } from "../components/bits";
import { useEngine } from "../engine";
import { SortTable, type Col } from "../components/SortTable";

interface WeekRow {
  id: string;
  name: string;
  position: string;
  team: string;
  opp: string;
  is_home: boolean;
  matchup: number;
  vegas: number;
  avail: number;
  pts: number;
  pass_yd: number;
  pass_td: number;
  rush_yd: number;
  rush_td: number;
  rec: number;
  rec_yd: number;
  rec_td: number;
  injury_status: string | null;
  headshot: string | null;
  floor?: number;
  ceiling?: number;
}

export default function Rankings() {
  const meta = useApi<Meta>("/api/meta");
  const { ep, isModel } = useEngine();
  const [view, setView] = useState<number | "ros">("ros");
  const [pos, setPos] = useState("ALL");
  const posQ = pos !== "ALL" ? `&pos=${pos}` : "";
  const ros = useApi<RankedPlayer[]>(view === "ros" ? ep(`/api/rankings?limit=600${posQ}`) : null);
  const week = useApi<WeekRow[]>(view !== "ros" ? ep(`/api/projections/week/${view}?limit=600${posQ}`) : null);

  const rosCols: Col<RankedPlayer>[] = [
    { key: "rank", label: "#", num: true, sort: (r) => r.overall_rank, render: (r) => r.overall_rank },
    { key: "name", label: "Player", sort: (r) => r.name, render: (r) => <PlayerCell {...r} /> },
    { key: "pr", label: "Pos rk", num: true, sort: (r) => r.pos_rank, render: (r) => `${r.position}${r.pos_rank}` },
    { key: "value", label: "Value", num: true, title: "Rest-of-season points over replacement", sort: (r) => r.value, render: (r) => <b>{fmt(r.value, 0)}</b> },
    { key: "ros", label: "ROS pts", num: true, sort: (r) => r.ros_pts, render: (r) => fmt(r.ros_pts, 0) },
    { key: "ppg", label: "Proj PPG", num: true, sort: (r) => r.proj_ppg, render: (r) => fmt(r.proj_ppg) },
    ...(isModel
      ? [{ key: "range", label: "Weekly range", num: true, title: "Typical week: 80% of games land between floor and ceiling", sort: (r: RankedPlayer) => r.ceiling_ppg, render: (r: RankedPlayer) => <Range floor={r.floor_ppg} ceiling={r.ceiling_ppg} /> }]
      : []),
    { key: "gl", label: "Games left", num: true, sort: (r) => r.games_left, render: (r) => r.games_left },
    { key: "next", label: "Next", sort: (r) => r.next_opp, render: (r) => r.next_opp ? `W${r.next_week} ${r.next_opp}` : "–" },
    { key: "np", label: "Next pts", num: true, sort: (r) => r.next_pts, render: (r) => fmt(r.next_pts) },
    { key: "po", label: "Playoff wks", num: true, title: "Opponent factor for weeks 15-17, the fantasy playoffs. Above 1 is easier.", sort: (r) => r.playoff_sos ?? null, render: (r) => <span style={{ color: (r.playoff_sos ?? 1) > 1.04 ? "var(--good)" : (r.playoff_sos ?? 1) < 0.96 ? "var(--bad)" : undefined }}>{fmt(r.playoff_sos, 2)}</span> },
    { key: "sos", label: "Schedule", num: true, title: "Average opponent factor for remaining games. Above 1 is easier.", sort: (r) => r.sos, render: (r) => <span style={{ color: r.sos > 1.03 ? "var(--good)" : r.sos < 0.97 ? "var(--bad)" : undefined }}>{fmt(r.sos, 2)}</span> },
  ];

  const weekCols: Col<WeekRow>[] = [
    { key: "name", label: "Player", sort: (r) => r.name, render: (r) => <PlayerCell {...r} /> },
    { key: "opp", label: "Opp", sort: (r) => r.opp, render: (r) => `${r.is_home ? "vs" : "@"} ${r.opp}` },
    { key: "pts", label: "Proj pts", num: true, sort: (r) => r.pts, render: (r) => <b>{fmt(r.pts)}</b> },
    ...(isModel
      ? [
          { key: "floor", label: "Floor", num: true, title: "Only 10% of the time below this", sort: (r: WeekRow) => r.floor, render: (r: WeekRow) => fmt(r.floor) },
          { key: "ceil", label: "Ceiling", num: true, title: "Only 10% of the time above this", sort: (r: WeekRow) => r.ceiling, render: (r: WeekRow) => fmt(r.ceiling) },
        ]
      : []),
    { key: "m", label: "Matchup", num: true, title: "Opponent factor for this position (above 1 is easier)", sort: (r) => r.matchup, render: (r) => <span style={{ color: r.matchup > 1.04 ? "var(--good)" : r.matchup < 0.96 ? "var(--bad)" : undefined }}>{fmt(r.matchup, 2)}</span> },
    { key: "v", label: "Vegas", num: true, title: "Implied team total vs this team's average (when lines are posted)", sort: (r) => r.vegas, render: (r) => fmt(r.vegas, 2) },
    { key: "a", label: "Plays", num: true, sort: (r) => r.avail, render: (r) => pct(r.avail) },
    { key: "py", label: "Pass yds", num: true, sort: (r) => r.pass_yd, render: (r) => fmt(r.pass_yd, 0) },
    { key: "ry", label: "Rush yds", num: true, sort: (r) => r.rush_yd, render: (r) => fmt(r.rush_yd, 0) },
    { key: "rec", label: "Rec", num: true, sort: (r) => r.rec, render: (r) => fmt(r.rec) },
    { key: "cy", label: "Rec yds", num: true, sort: (r) => r.rec_yd, render: (r) => fmt(r.rec_yd, 0) },
    { key: "td", label: "TDs", num: true, sort: (r) => r.pass_td + r.rush_td + r.rec_td, render: (r) => fmt(r.pass_td + r.rush_td + r.rec_td, 2) },
  ];

  const err = meta.error || ros.error || week.error;
  return (
    <div className="stack">
      <div className="page-head">
        <div>
          <h1>{isModel ? "Rankings, model edition" : "Rankings and projections"}</h1>
          <p>
            {isModel
              ? "Projections from the trained model blended with Sleeper, adjusted for injuries. Floor and ceiling show the range a player lands in 80% of weeks."
              : "Projections blend this app's usage model with Sleeper's, then adjust for matchup, Vegas lines and injuries."}
          </p>
        </div>
      </div>
      <div className="row">
        <select className="input" value={String(view)} onChange={(e) => setView(e.target.value === "ros" ? "ros" : Number(e.target.value))} aria-label="Timeframe">
          <option value="ros">Rest of season</option>
          {meta.data?.projection_weeks.map((w) => (
            <option key={w} value={w}>
              Week {w}
            </option>
          ))}
        </select>
        <PosFilter value={pos} onChange={setPos} />
      </div>
      <ErrorBox error={err} />
      {view === "ros" ? (
        ros.data ? <SortTable key={String(isModel)} rows={ros.data} cols={rosCols} rowKey={(r) => r.id} initial={{ key: "value", desc: true }} limit={300} /> : <Loading />
      ) : week.data ? (
        <SortTable key={view} rows={week.data} cols={weekCols} rowKey={(r) => r.id} initial={{ key: "pts", desc: true }} limit={300} />
      ) : (
        <Loading />
      )}
    </div>
  );
}
