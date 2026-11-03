import { useMemo, useState } from "react";
import { Link } from "react-router-dom";
import {
  api, fmt, odds, useApi, type Acceptance, type Advice, type Call, type League, type Settings, type TeamOdds,
} from "../api";
import { OddsChange, SituationBadge } from "../components/OddsBits";
import { CallBadge, ErrorBox, Headshot, InjuryBadge, Loading, PosFilter, Range } from "../components/bits";
import { useEngine } from "../engine";

const TABS: { call: Call; label: string; blurb: string }[] = [
  { call: "BUY LOW", label: "Buy low", blurb: "Projected to score more than managers are seeing. Trade for them now." },
  { call: "SELL HIGH", label: "Sell high", blurb: "Producing above what usage supports. Trade them while their price is up." },
  { call: "HOLD", label: "Hold", blurb: "Production matches the projection. Priced fairly, so no edge either way." },
];

interface MyOdds {
  playoff: number;
  title: number;
  situation: string;
}

interface TradeIdea {
  team_id: string;
  team_name: string;
  give: { id: string; name: string; position: string; value: number; call: Call | null }[];
  get: { id: string; name: string; position: string; value: number; call: Call | null }[];
  my_gain: number;
  their_gain: number;
  odds: { a: TeamOdds; b: TeamOdds };
  acceptance: Acceptance;
}

export default function Advisor() {
  const [tab, setTab] = useState<Call>("BUY LOW");
  const [pos, setPos] = useState("ALL");
  const [scope, setScope] = useState<"all" | "mine" | "others">("all");
  const { ep, base, isModel } = useEngine();
  const advice = useApi<Advice[]>(ep("/api/advice"));
  const settings = useApi<Settings>("/api/settings");
  const hasLeague = !!settings.data?.espn.league_id;
  const league = useApi<League>(hasLeague ? ep("/api/league") : null);
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
          <h1>{isModel ? "Trade advisor, model edition" : "Trade advisor"}</h1>
          <p>
            Trade partners judge players by recent points. These calls compare that against what{" "}
            {isModel ? "the trained model" : "usage, efficiency and the remaining schedule"} project from here.
          </p>
        </div>
      </div>

      {myTeam && <TradeIdeas teamName={myTeam.name} />}
      {myTeam && <WaiverPickups />}

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
                <Link to={`${base}/player/${encodeURIComponent(r.id)}`}>
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
              {r.floor_ppg !== undefined && (
                <div className="stat">
                  <b><Range floor={r.floor_ppg} ceiling={r.ceiling_ppg} /></b>
                  <span>Range</span>
                </div>
              )}
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

interface Pickup {
  add: { id: string; name: string; position: string; team: string; value: number; proj_ppg: number; headshot: string | null };
  drop: { id: string; name: string; position: string; value: number; proj_ppg: number };
  lineup_gain: number;
  playoff_before: number;
  playoff_after: number;
  title_before: number;
  title_after: number;
  status: string | null;
  pct_owned: number | null;
}

function WaiverPickups() {
  const { ep, base } = useEngine();
  const [rows, setRows] = useState<Pickup[] | null>(null);
  const [checked, setChecked] = useState(0);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const run = async () => {
    setBusy(true);
    setError(null);
    try {
      const r = await api<{ options: Pickup[]; free_agents_checked: number }>(ep("/api/league/waivers"));
      setRows(r.options);
      setChecked(r.free_agents_checked);
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
          <h2>Waiver pickups</h2>
          <p className="small muted">
            Sometimes the best trade is free. Checks every available player in your league against your roster, picks
            the drop that hurts least (never your IR slot), and simulates the season with the pickup.
          </p>
        </div>
        <button className="btn ghost" onClick={run} disabled={busy}>
          {busy ? "Checking waivers…" : rows ? "Check again" : "Find pickups"}
        </button>
      </div>
      <ErrorBox error={error} />
      {rows && !rows.length && <div className="empty">No available player would improve your lineup right now.</div>}
      {rows && rows.length > 0 && (
        <>
          <div className="tablewrap">
            <table>
              <thead>
                <tr>
                  <th>Add</th>
                  <th>Drop</th>
                  <th className="n" title="Projected PPG rest of season">Proj PPG</th>
                  <th className="n" title="Change in your projected starting lineup, rest of season">Lineup gain</th>
                  <th className="n">Your playoff odds</th>
                  <th className="n">Your title odds</th>
                  <th>Status</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((r) => (
                  <tr key={r.add.id}>
                    <td>
                      <span className="player-cell">
                        <Headshot src={r.add.headshot} />
                        <Link to={`${base}/player/${encodeURIComponent(r.add.id)}`}>{r.add.name}</Link>
                        <span className="pos">{r.add.position} {r.add.team}</span>
                      </span>
                    </td>
                    <td>{r.drop.name} <span className="pos">{r.drop.position}</span></td>
                    <td className="n">{fmt(r.add.proj_ppg)} <span className="muted small">vs {fmt(r.drop.proj_ppg)}</span></td>
                    <td className="n delta up">+{fmt(r.lineup_gain)}</td>
                    <td className="n"><OddsChange before={r.playoff_before} after={r.playoff_after} /></td>
                    <td className="n"><OddsChange before={r.title_before} after={r.title_after} /></td>
                    <td className="small">
                      {r.status === "WAIVERS" ? "On waivers" : "Free agent"}
                      {r.pct_owned !== null && <div className="muted">{Math.round(r.pct_owned)}% owned on ESPN</div>}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="small muted">{checked} available players checked.</p>
        </>
      )}
    </section>
  );
}

function TradeIdeas({ teamName }: { teamName: string }) {
  const { ep, base } = useEngine();
  const [ideas, setIdeas] = useState<TradeIdea[] | null>(null);
  const [myOdds, setMyOdds] = useState<MyOdds | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const run = async () => {
    setBusy(true);
    setError(null);
    try {
      const r = await api<{ trades: TradeIdea[]; my_odds: MyOdds }>(ep("/api/league/trades"));
      setIdeas(r.trades);
      setMyOdds(r.my_odds);
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
            Searches 1-for-1 and 2-for-1 deals with every team, simulates the rest of the season before and after each,
            and ranks them by how much they raise your playoff and title odds, weighted by how appealing the offer
            looks to the other manager.
          </p>
          {myOdds && (
            <p className="small">
              Right now: <b>{odds(myOdds.playoff)}</b> playoff odds, <b>{odds(myOdds.title)}</b> title odds{" "}
              <SituationBadge s={myOdds.situation} />
            </p>
          )}
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
                <th className="n" title="Your chance of making the playoffs, before and after">Your playoff odds</th>
                <th className="n" title="Your chance of winning the league, before and after">Your title odds</th>
                <th className="n" title="Change in their playoff odds">Their playoff odds</th>
                <th title="How the offer looks to them, and an estimated chance they accept">Would they accept?</th>
                <th></th>
              </tr>
            </thead>
            <tbody>
              {ideas.map((t, i) => {
                const d = t.odds.b.playoff_delta * 100;
                const acls = t.acceptance.label === "Strong offer" ? "buy" : t.acceptance.label === "Long shot" ? "sell" : "hold";
                return (
                  <tr key={i}>
                    <td>{t.give.map((p) => <div key={p.id}>{p.name} <span className="pos">{p.position}</span> <CallBadge call={p.call} /></div>)}</td>
                    <td>{t.get.map((p) => <div key={p.id}>{p.name} <span className="pos">{p.position}</span> <CallBadge call={p.call} /></div>)}</td>
                    <td>
                      <div>{t.team_name}</div>
                      <SituationBadge s={t.odds.b.situation} />
                    </td>
                    <td className="n"><OddsChange before={t.odds.a.playoff_before} after={t.odds.a.playoff_after} /></td>
                    <td className="n"><OddsChange before={t.odds.a.title_before} after={t.odds.a.title_after} /></td>
                    <td className={`n delta ${d > 0.5 ? "up" : d < -0.5 ? "down" : ""}`}>{d > 0 ? "+" : ""}{fmt(d, 1)} pts</td>
                    <td className="small" title={t.acceptance.reasons.join("\n")}>
                      <span className={`badge ${acls}`}>{t.acceptance.label}</span>
                      <div className="muted">~{Math.round(t.acceptance.probability * 100)}% chance</div>
                    </td>
                  <td>
                    <Link
                      className="btn ghost small"
                      to={`${base ? `${base}/trade` : "/"}?a=${t.give.map((p) => p.id).join(",")}&b=${t.get.map((p) => p.id).join(",")}&tb=${t.team_id}`}
                    >
                      Open in calculator
                    </Link>
                  </td>
                </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}
