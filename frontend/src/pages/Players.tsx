import { useState } from "react";
import { fmt, pct, useApi, type Meta } from "../api";
import { ErrorBox, Loading, PlayerCell, PosFilter } from "../components/bits";
import { SortTable, type Col } from "../components/SortTable";

interface Row {
  id: string;
  name: string;
  position: string;
  team: string | null;
  headshot: string | null;
  injury_status: string | null;
  value: number | null;
  ros_pts: number | null;
  proj_ppg: number | null;
  games: number | null;
  pts: number | null;
  ppg: number | null;
  xfp_ppg: number | null;
  last3_ppg: number | null;
  target_share: number | null;
  wopr: number | null;
  tgt_pg: number | null;
  car_pg: number | null;
  td: number | null;
  xtd: number | null;
  pass_yd: number | null;
  rush_yd: number | null;
  rec_yd: number | null;
}

export default function Players() {
  const meta = useApi<Meta>("/api/meta");
  const [season, setSeason] = useState<number | null>(null);
  const [pos, setPos] = useState("ALL");
  const [q, setQ] = useState("");
  const s = season ?? meta.data?.season;
  const path = s
    ? `/api/players?season=${s}&limit=2000${pos !== "ALL" ? `&pos=${pos}` : ""}`
    : null;
  const data = useApi<Row[]>(path);
  const isCurrent = s === meta.data?.season;
  const rows = (data.data ?? []).filter((r) => !q || r.name.toLowerCase().includes(q.toLowerCase()));

  const cols: Col<Row>[] = [
    { key: "name", label: "Player", sort: (r) => r.name, render: (r) => <PlayerCell {...r} /> },
    ...(isCurrent
      ? [
          { key: "value", label: "Value", num: true, title: "Rest-of-season points over replacement", sort: (r: Row) => r.value, render: (r: Row) => fmt(r.value, 0) },
          { key: "proj", label: "Proj PPG", num: true, sort: (r: Row) => r.proj_ppg, render: (r: Row) => fmt(r.proj_ppg) },
        ]
      : []),
    { key: "games", label: "GP", num: true, sort: (r) => r.games, render: (r) => r.games ?? "–" },
    { key: "pts", label: "Pts", num: true, sort: (r) => r.pts, render: (r) => fmt(r.pts) },
    { key: "ppg", label: "PPG", num: true, sort: (r) => r.ppg, render: (r) => fmt(r.ppg) },
    { key: "xfp", label: "xFP/G", num: true, title: "Expected points per game from usage at league-average efficiency", sort: (r) => r.xfp_ppg, render: (r) => fmt(r.xfp_ppg) },
    { key: "last3", label: "Last 3", num: true, sort: (r) => r.last3_ppg, render: (r) => fmt(r.last3_ppg) },
    { key: "ts", label: "Tgt share", num: true, sort: (r) => r.target_share, render: (r) => pct(r.target_share) },
    { key: "wopr", label: "WOPR", num: true, title: "Weighted opportunity rating: target share and air-yard share combined", sort: (r) => r.wopr, render: (r) => fmt(r.wopr, 2) },
    { key: "tgt", label: "Tgt/G", num: true, sort: (r) => r.tgt_pg, render: (r) => fmt(r.tgt_pg) },
    { key: "car", label: "Car/G", num: true, sort: (r) => r.car_pg, render: (r) => fmt(r.car_pg) },
    { key: "td", label: "TD", num: true, sort: (r) => r.td, render: (r) => r.td ?? "–" },
    { key: "xtd", label: "xTD", num: true, title: "Touchdowns expected from usage", sort: (r) => r.xtd, render: (r) => fmt(r.xtd) },
    { key: "pyd", label: "Pass yds", num: true, sort: (r) => r.pass_yd, render: (r) => r.pass_yd ?? "–" },
    { key: "ryd", label: "Rush yds", num: true, sort: (r) => r.rush_yd, render: (r) => r.rush_yd ?? "–" },
    { key: "cyd", label: "Rec yds", num: true, sort: (r) => r.rec_yd, render: (r) => r.rec_yd ?? "–" },
  ];

  return (
    <div className="stack">
      <div className="page-head">
        <div>
          <h1>Player stats</h1>
          <p>Scored with your league settings. xFP shows what a player's opportunities are usually worth.</p>
        </div>
      </div>
      <div className="row">
        <select className="input" value={s ?? ""} onChange={(e) => setSeason(Number(e.target.value))} aria-label="Season">
          {meta.data?.seasons.map((x) => (
            <option key={x} value={x}>
              {x} season
            </option>
          ))}
        </select>
        <PosFilter value={pos} onChange={setPos} />
        <input className="input grow" placeholder="Filter by name" value={q} onChange={(e) => setQ(e.target.value)} aria-label="Filter by name" />
      </div>
      <ErrorBox error={data.error || meta.error} />
      {data.loading && !data.data ? (
        <Loading what="Loading stats" />
      ) : (
        <SortTable
          key={`${s}-${isCurrent}`}
          rows={rows}
          cols={cols}
          rowKey={(r) => r.id}
          initial={{ key: isCurrent ? "value" : "pts", desc: true }}
          limit={400}
        />
      )}
    </div>
  );
}
