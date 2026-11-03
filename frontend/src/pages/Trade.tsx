import { useEffect, useMemo, useState } from "react";
import { useSearchParams } from "react-router-dom";
import {
  api, fmt, useApi, type Acceptance, type Call, type League, type Meta, type RankedPlayer, type Settings,
  type TeamOdds,
} from "../api";
import { AcceptanceBox, TeamOddsCard } from "../components/OddsBits";
import { CallBadge, ErrorBox, Headshot } from "../components/bits";
import { GameLog } from "../components/GameLog";
import { PlayerSearch, type SearchItem } from "../components/PlayerSearch";
import { YardField } from "../components/YardField";
import { useEngine } from "../engine";

interface TradePlayer {
  id: string;
  name: string;
  position: string | null;
  team: string | null;
  value: number;
  ros_pts: number;
  proj_ppg: number;
  call?: Call;
}
interface TradeResult {
  side_a: { gives: TradePlayer[]; receives_value: number; raw_receives: number };
  side_b: { gives: TradePlayer[]; receives_value: number; raw_receives: number };
  verdict: { label: string; winner: "A" | "B" | null; diff_pct: number };
  balance?: { needed_value: number; add_to: "A" | "B"; suggestions: TradePlayer[] };
  lineup?: {
    weeks: number;
    a: { before: number; after: number; delta: number; per_week: number };
    b: { before: number; after: number; delta: number; per_week: number };
  };
  odds?: { a: TeamOdds; b: TeamOdds };
  acceptance?: Acceptance;
  teams?: { a: string; b: string };
}

export default function Trade() {
  const [params, setParams] = useSearchParams();
  const { ep, isModel } = useEngine();
  const ranked = useApi<RankedPlayer[]>(ep("/api/rankings?limit=800"));
  const settings = useApi<Settings>("/api/settings");
  const hasLeague = !!settings.data?.espn.league_id;
  const league = useApi<League>(hasLeague ? ep("/api/league") : null);

  const split = (k: string) => (params.get(k) || "").split(",").filter(Boolean);
  const [a, setA] = useState<string[]>(() => split("a"));
  const [b, setB] = useState<string[]>(() => split("b"));
  const [teamA, setTeamA] = useState(params.get("ta") || "");
  const [teamB, setTeamB] = useState(params.get("tb") || "");
  const [result, setResult] = useState<TradeResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const meta = useApi<Meta>("/api/meta");
  const [openLogs, setOpenLogs] = useState<Set<string>>(() => new Set(split("logs")));
  const toggleLog = (id: string) =>
    setOpenLogs((s) => {
      const n = new Set(s);
      if (n.has(id)) n.delete(id);
      else n.add(id);
      return n;
    });

  // Default side A to my ESPN team once the league loads.
  useEffect(() => {
    const mine = league.data?.my_team_id;
    if (mine && !teamA && !params.get("ta")) setTeamA(mine);
  }, [league.data, teamA, params]);

  const byId = useMemo(() => {
    const m = new Map<string, SearchItem & { value: number }>();
    ranked.data?.forEach((p) => m.set(p.id, p));
    return m;
  }, [ranked.data]);

  const rosterItems = (teamId: string): SearchItem[] => {
    const t = league.data?.teams.find((x) => x.team_id === teamId);
    if (!t) return ranked.data ?? [];
    return t.roster
      .filter((p) => p.id)
      .map((p) => byId.get(p.id!) ?? { id: p.id!, name: p.name, position: p.position, team: p.pro_team, value: 0 });
  };

  useEffect(() => {
    setParams(
      (prev) => {
        const n = new URLSearchParams(prev);
        a.length ? n.set("a", a.join(",")) : n.delete("a");
        b.length ? n.set("b", b.join(",")) : n.delete("b");
        teamA ? n.set("ta", teamA) : n.delete("ta");
        teamB ? n.set("tb", teamB) : n.delete("tb");
        return n;
      },
      { replace: true },
    );
    if (!a.length || !b.length) {
      setResult(null);
      return;
    }
    let cancelled = false;
    setError(null);
    api<TradeResult>(ep("/api/trade"), {
      method: "POST",
      body: JSON.stringify({
        side_a: a,
        side_b: b,
        team_a: teamA && teamB ? teamA : null,
        team_b: teamA && teamB ? teamB : null,
      }),
    })
      .then((r) => !cancelled && setResult(r))
      .catch((e: Error) => !cancelled && setError(e.message));
    return () => {
      cancelled = true;
    };
    // `ep` is rebuilt every render; isModel captures the only thing it depends on.
  }, [a, b, teamA, teamB, setParams, isModel]);

  const teamName = (id: string, fallback: string) =>
    league.data?.teams.find((t) => t.team_id === id)?.name || fallback;
  // Teams can also be detected from whose players are in the trade.
  const nameA = teamName(teamA || result?.teams?.a || "", "Side A");
  const nameB = teamName(teamB || result?.teams?.b || "", "Side B");
  const exclude = new Set([...a, ...b]);
  const callOf = (id: string, side: TradePlayer[] | undefined) => side?.find((p) => p.id === id)?.call;

  const sidePanel = (
    label: string,
    ids: string[],
    set: (v: string[]) => void,
    team: string,
    setTeam: (v: string) => void,
    resultSide?: TradePlayer[],
  ) => (
    <section className="panel">
      <div className="side-head">
        <h2>{label} gives</h2>
        {league.data && (
          <select className="input" value={team} onChange={(e) => setTeam(e.target.value)} aria-label={`${label} team`}>
            <option value="">Any player</option>
            {league.data.teams.map((t) => (
              <option key={t.team_id} value={t.team_id}>
                {t.name}
              </option>
            ))}
          </select>
        )}
      </div>
      <PlayerSearch
        items={rosterItems(team)}
        exclude={exclude}
        onPick={(p) => set([...ids, p.id])}
        placeholder={team ? "Add from this roster" : "Search any player"}
      />
      <div className="chips">
        {ids.map((id) => {
          const p = byId.get(id);
          const open = openLogs.has(id);
          return (
            <div key={id} className="chip-wrap">
              <div className="chip">
                <Headshot src={p?.headshot} />
                <span>
                  <b>{p?.name ?? id}</b>{" "}
                  <span className="pos">
                    {p?.position} {p?.team ?? ""}
                  </span>
                </span>
                <span className="row">
                  <button
                    className="linkbtn small"
                    aria-expanded={open}
                    onClick={() => toggleLog(id)}
                  >
                    {open ? "Hide game log" : "Game log"}
                  </button>
                  <CallBadge call={callOf(id, resultSide)} />
                  <span className="num" title="Value: rest-of-season points over a replacement-level player">
                    {fmt(p?.value, 0)}
                  </span>
                </span>
                <button className="x" aria-label={`Remove ${p?.name ?? id}`} onClick={() => set(ids.filter((x) => x !== id))}>
                  ×
                </button>
              </div>
              {open && <GameLog id={id} lastWeek={meta.data?.last_stats_week} />}
            </div>
          );
        })}
        {!ids.length && <div className="muted small">No players yet.</div>}
      </div>
    </section>
  );

  const v = result?.verdict;
  const headline = !v
    ? null
    : v.winner === null
      ? "Fair trade"
      : `${v.winner === "A" ? nameA : nameB} wins`;

  return (
    <div className="stack">
      <div className="page-head">
        <div>
          <h1>{isModel ? "Trade calculator, model edition" : "Trade calculator"}</h1>
          <p>
            {isModel
              ? "Same calculator, but every value comes from the trained model blended with Sleeper. "
              : ""}
            Values are rest-of-season points above a replacement-level player in your league format. Pick
            ESPN teams to also see how each starting lineup changes.
          </p>
        </div>
        {(a.length > 0 || b.length > 0) && (
          <button className="btn ghost" onClick={() => { setA([]); setB([]); }}>
            Clear trade
          </button>
        )}
      </div>
      <ErrorBox error={ranked.error || error} />
      <div className="trade-sides">
        {sidePanel(nameA, a, setA, teamA, setTeamA, result?.side_a.gives)}
        {sidePanel(nameB, b, setB, teamB, setTeamB, result?.side_b.gives)}
      </div>

      {result && v ? (
        <section className="panel stack">
          <YardField aLabel={nameA} bLabel={nameB} aGets={result.side_a.receives_value} bGets={result.side_b.receives_value} />
          <div className="verdict">
            <span className="big">{headline}</span>
            <span className="muted">
              {v.label}
              {v.winner ? `, by ${Math.abs(v.diff_pct).toFixed(0)}% of value` : ""}
            </span>
          </div>
          <div className="row" style={{ gap: 32 }}>
            <div className="stat">
              <b>{fmt(result.side_a.receives_value)}</b>
              <span>{nameA} receives</span>
            </div>
            <div className="stat">
              <b>{fmt(result.side_b.receives_value)}</b>
              <span>{nameB} receives</span>
            </div>
            {result.lineup && (
              <>
                <LineupDelta label={`${nameA} lineup`} d={result.lineup.a} />
                <LineupDelta label={`${nameB} lineup`} d={result.lineup.b} />
              </>
            )}
          </div>
          {result.odds && (
            <div className="stack" style={{ gap: 10 }}>
              <h3>What it does to each team's season</h3>
              <div className="odds-grid">
                <TeamOddsCard o={result.odds.a} label={nameA} />
                <TeamOddsCard o={result.odds.b} label={nameB} />
                {result.acceptance && <AcceptanceBox a={result.acceptance} teamName={nameB} />}
              </div>
              <p className="small muted">
                From simulating the rest of the season thousands of times with each team's best weekly lineup,
                the real remaining schedule and playoff bracket.
              </p>
            </div>
          )}
          <p className="small muted">
            Receiving values include a consolidation adjustment: two good players are worth less than their sum,
            because they cost an extra roster spot and you can only start so many.
            {result.lineup &&
              ` Lineup change is the projected best starting lineup over the remaining ${result.lineup.weeks} weeks, before and after.`}
          </p>
          {result.balance && (
            <div>
              <h3>Even it out</h3>
              <p className="small muted">
                {result.balance.add_to === "A" ? nameA : nameB} could add about {fmt(result.balance.needed_value, 0)} in
                value. Players near that value:
              </p>
              <div className="row" style={{ marginTop: 8 }}>
                {result.balance.suggestions.map((s) => (
                  <button
                    key={s.id}
                    className="btn ghost"
                    onClick={() => (result.balance!.add_to === "A" ? setA([...a, s.id]) : setB([...b, s.id]))}
                  >
                    + {s.name} <span className="pos">{s.position}</span> {fmt(s.value, 0)}
                  </button>
                ))}
              </div>
            </div>
          )}
        </section>
      ) : (
        <div className="empty">Add at least one player to each side to see who wins the trade.</div>
      )}
    </div>
  );
}

function LineupDelta({ label, d }: { label: string; d: { delta: number; per_week: number } }) {
  const cls = d.delta > 0.5 ? "up" : d.delta < -0.5 ? "down" : "";
  return (
    <div className="stat">
      <b className={`delta ${cls}`}>
        {d.delta > 0 ? "+" : ""}
        {fmt(d.delta)}
      </b>
      <span>
        {label}, {d.per_week > 0 ? "+" : ""}
        {fmt(d.per_week, 2)} pts/week
      </span>
    </div>
  );
}
