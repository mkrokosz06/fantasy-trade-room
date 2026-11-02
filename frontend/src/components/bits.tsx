import { Link } from "react-router-dom";
import type { Call } from "../api";
import { useEngine } from "../engine";

export function CallBadge({ call }: { call?: Call | null }) {
  if (!call) return null;
  const cls = call === "BUY LOW" ? "buy" : call === "SELL HIGH" ? "sell" : "hold";
  const label = call === "BUY LOW" ? "Buy low" : call === "SELL HIGH" ? "Sell high" : "Hold";
  return <span className={`badge ${cls}`}>{label}</span>;
}

export function InjuryBadge({ status }: { status?: string | null }) {
  if (!status) return null;
  return <span className="badge inj" title="Injury status from Sleeper">{status}</span>;
}

export function Headshot({ src, size }: { src?: string | null; size?: number }) {
  return (
    <img
      className="headshot"
      src={src || undefined}
      alt=""
      loading="lazy"
      style={size ? { width: size, height: size } : undefined}
      onError={(e) => ((e.target as HTMLImageElement).style.visibility = "hidden")}
    />
  );
}

export function PlayerCell(p: {
  id: string;
  name: string;
  position?: string | null;
  team?: string | null;
  headshot?: string | null;
  injury_status?: string | null;
}) {
  const { base } = useEngine();
  return (
    <span className="player-cell">
      <Headshot src={p.headshot} />
      <Link to={`${base}/player/${encodeURIComponent(p.id)}`}>{p.name}</Link>
      <span className="pos">
        {p.position} {p.team ?? "FA"}
      </span>
      <InjuryBadge status={p.injury_status} />
    </span>
  );
}

export function Loading({ what = "Loading" }: { what?: string }) {
  return <div className="loading">{what}…</div>;
}

export function ErrorBox({ error }: { error: string | null }) {
  if (!error) return null;
  const untrained = error.includes("hasn't been trained");
  return (
    <div className="error" role="alert">
      {error} {untrained && <Link to="/model">Go to Model lab</Link>}
    </div>
  );
}

/** Floor-ceiling range: where the player lands 80% of the time. */
export function Range({ floor, ceiling }: { floor?: number | null; ceiling?: number | null }) {
  if (floor === null || floor === undefined || ceiling === null || ceiling === undefined) return <>–</>;
  return (
    <span className="range" title="80% of the time, the player scores between these">
      {floor.toFixed(1)}–{ceiling.toFixed(1)}
    </span>
  );
}

const POSITIONS = ["ALL", "QB", "RB", "WR", "TE", "K"];

export function PosFilter({
  value,
  onChange,
  withK = true,
}: {
  value: string;
  onChange: (v: string) => void;
  withK?: boolean;
}) {
  return (
    <div className="seg" role="group" aria-label="Position">
      {POSITIONS.filter((p) => withK || p !== "K").map((p) => (
        <button key={p} aria-pressed={value === p} onClick={() => onChange(p)}>
          {p === "ALL" ? "All" : p}
        </button>
      ))}
    </div>
  );
}
