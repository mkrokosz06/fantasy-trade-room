import { useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { api, fmt, useApi, type Advice, type Call, type League, type Settings } from "../api";
import { CallBadge, ErrorBox, Headshot, InjuryBadge, Loading, PosFilter } from "../components/bits";

const TABS: { call: Call; label: string; blurb: string }[] = [
  { call: "BUY LOW", label: "Buy low", blurb: "Projected to score more than managers are seeing. Trade for them now." },
  { call: "SELL HIGH", label: "Sell high", blurb: "Producing above what usage supports. Trade them while their price is up." },
  { call: "HOLD", label: "Hold", blurb: "Production matches the projection. Priced fairly, so no edge either way." },
];

interface TradeIdea {
  team_id: string;
  team_name: string;
  give: { id: string; name: string; position: string; value: number; call: Call | null }[];
  get: { id: string; name: string; position: string; value: number; call: Call | null }[];
  my_gain: number;
  their_gain: number;
  looks_fair_to_them: boolean;
}

export default function Advisor() {
  const [tab, setTab] = useState<Call>("BUY LOW");
  const [pos, setPos] = useState("ALL");
  const [scope, setScope] = useState<"all" | "mine" | "others">("all");
  const advice = useApi<Advice[]>("/api/advice");
  const settings = useApi<Settings>("/api/settings");
  const hasLeague = !!settings.data?.espn.league_id;
  const league = useApi<League>(hasLeague ? "/api/league" : null);
  const myTeam = league.data?.teams.find((t) => t.team_id === league.data?.my_team_id);
  const myIds = useMemo(() => new Set(myTeam?.roster.map((p) => p.id).filter(Boolean) as string[]), [myTeam]);
  const ownerOf = useMemo(() => {
    const m = new Map<string, string>();
    league.data?.teams.forEach((t) => t.roster.forEach((p) => p.id && m.set(p.id, t.name)));
    return m;
  }, [league.data]);

  const rows = useMemo(() => {
    let r = (advice.data ?? []).filter((x) => x.call === tab);
    if (pos !== "ALL") r = r.filter((x) => x.position === pos);
    if (myTeam && scope === "mine") r = r.filter((x) => myIds.has(x.id));
    if (myTeam && scope === "others") r = r.filter((x) => !myIds.has(x.id) && ownerOf.has(x.id));
    return [...r].sort((x, y) => y.confidence - x.confidence);
  }, [advice.data, tab, pos, scope, myTeam, myIds, ownerOf]);

  const counts = useMemo(() => {
    const c: Record<string, number> = {};
    (advice.data ?? []).forEach((x) => (c[x.call] = (c[x.call] ?? 0) + 1));
    return c;
  }, [advice.data]);

  return (
    <div className="stack">
      <div className="page-head">
        <div>
          <h1>Trade advisor</h1>
          <p>
            Trade partners judge players by recent points. These calls compare that against what usage, efficiency
            and the remaining schedule project from here.
          </p>
        </div>
      </div>

      {myTeam && <TradeIdeas teamName={myTeam.name} />}

      <div className="row">
        <div className="seg" role="tablist" aria-label="Recommendation">
          {TABS.map((t) => (
            <button key={t.call} role="tab" aria-pressed={tab === t.call} aria-selected={tab === t.call} onClick={() => setTab(t.call)}>
              {t.label} {counts[t.call] !== undefined && <span className="muted">({counts[t.call]})</span>}
            </button>
          ))}
        </div>
        <PosFilter value={pos} onChange={setPos} withK={false} />
        {myTeam && (
          <div className="seg" role="group" aria-label="Whose players">
            <button aria-pressed={scope === "all"} onClick={() => setScope("all")}>Everyone</button>
            <button aria-pressed={scope === "mine"} onClick={() => setScope("mine")}>My roster</button>
            <button aria-pressed={scope === "others"} onClick={() => setScope("others")}>Other rosters</button>
          </div>
        )}
      </div>
      <p className="muted">{TABS.find((t) => t.call === tab)!.blurb}</p>

      <ErrorBox error={advice.error} />
      {advice.loading && !advice.data && <Loading what="Grading players" />}
      <div className="advice-list">
        {rows.map((r) => (
          <article key={r.id} className={`advice ${tab === "BUY LOW" ? "buy" : tab === "SELL HIGH" ? "sell" : ""}`}>
            <Headshot src={r.headshot} />
            <div>
              <div className="row" style={{ gap: 8 }}>
                <Link to={`/player/${encodeURIComponent(r.id)}`}>
                  <b>{r.name}</b>
                </Link>
                <span className="pos">
                  {r.position} {r.team}
                </span>
                <CallBadge call={r.call} />
                <InjuryBadge status={r.injury_status} />
                {ownerOf.has(r.id) && (
                  <span className="small muted">
                    {!myIds.has(r.id)
                      ? `Owned by ${ownerOf.get(r.id)}`
                      : r.call === "BUY LOW"
                        ? "On your roster: keep, worth more than their recent points suggest"
                        : r.call === "SELL HIGH"
                          ? "On your roster: shop now while the price is high"
                          : "On your roster"}
                  </span>
                )}
              </div>
              <ul>
                {r.reasons.map((why, i) => (
                  <li key={i}>{why}</li>
                ))}
              </ul>
            </div>
            <div className="nums">
              <div className="stat">
                <b>{fmt(r.perceived_ppg)}</b>
                <span title="Half season average, half last 3 games: roughly how trade partners judge the player">Recent PPG</span>
              </div>
              <div className="stat">
                <b>{fmt(r.proj_ppg)}</b>
                <span>Projected</span>
              </div>
              <div className="stat" style={{ color: tab === "BUY LOW" ? "var(--buy)" : tab === "SELL HIGH" ? "var(--sell)" : "var(--hold)" }}>
                <b>{Math.round(r.confidence * 100)}</b>
                <span>Confidence</span>
              </div>
            </div>
          </article>
        ))}
        {advice.data && !rows.length && <div className="empty">No players match these filters.</div>}
      </div>
    </div>
  );
}

function TradeIdeas({ teamName }: { teamName: string }) {
  const [ideas, setIdeas] = useState<TradeIdea[] | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const run = async () => {
    setBusy(true);
    setError(null);
    try {
      const r = await api<{ trades: TradeIdea[] }>("/api/league/trades");
      setIdeas(r.trades);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <section className="panel stack">
      <div className="side-head">
        <div>
          <h2>Trade ideas for {teamName}</h2>
          <p className="small muted">
            Searches 1-for-1 and 2-for-1 deals with every team, keeping trades that raise your projected starting lineup.
            Ranked higher when the other team also improves or would see the deal as fair by recent points.
          </p>
        </div>
        <button className="btn" onClick={run} disabled={busy}>
          {busy ? "Searching trades…" : ideas ? "Search again" : "Find trades"}
        </button>
      </div>
      <ErrorBox error={error} />
      {ideas && !ideas.length && <div className="empty">No trades found that improve your lineup. Your roster may already be well balanced.</div>}
      {ideas && ideas.length > 0 && (
        <div className="tablewrap">
          <table>
            <thead>
              <tr>
                <th>You give</th>
                <th>You get</th>
                <th>From</th>
                <th className="n" title="Change in your projected starting-lineup points, rest of season">Your gain</th>
                <th className="n" title="Change in their projected starting-lineup points">Their change</th>
                <th>Likely to accept?</th>
                <th></th>
              </tr>
            </thead>
            <tbody>
              {ideas.map((t, i) => (
                <tr key={i}>
                  <td>{t.give.map((p) => <div key={p.id}>{p.name} <span className="pos">{p.position}</span> <CallBadge call={p.call} /></div>)}</td>
                  <td>{t.get.map((p) => <div key={p.id}>{p.name} <span className="pos">{p.position}</span> <CallBadge call={p.call} /></div>)}</td>
                  <td>{t.team_name}</td>
                  <td className="n delta up">+{fmt(t.my_gain)}</td>
                  <td className={`n delta ${t.their_gain > 0.5 ? "up" : t.their_gain < -0.5 ? "down" : ""}`}>
                    {t.their_gain > 0 ? "+" : ""}
                    {fmt(t.their_gain)}
                  </td>
                  <td className="small">
                    {t.their_gain > 0 && t.looks_fair_to_them
                      ? "Good odds: helps both teams"
                      : t.looks_fair_to_them
                        ? "Looks fair by recent points"
                        : t.their_gain > 0
                          ? "Helps them, but looks uneven"
                          : "Hard sell"}
                  </td>
                  <td>
                    <Link
                      className="btn ghost small"
                      to={`/?a=${t.give.map((p) => p.id).join(",")}&b=${t.get.map((p) => p.id).join(",")}&tb=${t.team_id}`}
                    >
                      Open in calculator
                    </Link>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}
