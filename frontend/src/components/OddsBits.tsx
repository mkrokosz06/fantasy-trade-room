import { odds, type Acceptance, type TeamOdds } from "../api";

export function SituationBadge({ s }: { s: string }) {
  const cls = s === "Contender" ? "buy" : s === "Long shot" ? "sell" : "hold";
  return <span className={`badge ${cls}`}>{s}</span>;
}

/** "6% → 14%" with the change colored. */
export function OddsChange({ before, after }: { before: number; after: number }) {
  const d = after - before;
  const cls = d > 0.004 ? "up" : d < -0.004 ? "down" : "";
  return (
    <span className="nowrap">
      {odds(before)} <span className="muted">→</span> <b className={`delta ${cls}`}>{odds(after)}</b>
    </span>
  );
}

export function TeamOddsCard({ o, label }: { o: TeamOdds; label: string }) {
  return (
    <div className="odds-card">
      <div className="row" style={{ gap: 8 }}>
        <b>{label}</b>
        <SituationBadge s={o.situation} />
      </div>
      <dl>
        <dt>Playoff odds</dt>
        <dd><OddsChange before={o.playoff_before} after={o.playoff_after} /></dd>
        <dt>Title odds</dt>
        <dd><OddsChange before={o.title_before} after={o.title_after} /></dd>
        <dt>First-round bye</dt>
        <dd><OddsChange before={o.bye_before} after={o.bye_after} /></dd>
        <dt>Projected wins</dt>
        <dd>
          {o.avg_wins_before.toFixed(1)} <span className="muted">→</span> <b>{o.avg_wins_after.toFixed(1)}</b>
        </dd>
      </dl>
      {o.must_drop && o.must_drop.length > 0 && (
        <p className="small muted" style={{ marginTop: 8 }}>
          Receives more players than it sends, so it would need to drop {o.must_drop.map((p) => p.name).join(" and ")}{" "}
          (its least valuable player{o.must_drop.length > 1 ? "s" : ""}). Odds include that.
        </p>
      )}
    </div>
  );
}

export function AcceptanceBox({ a, teamName }: { a: Acceptance; teamName: string }) {
  const cls = a.label === "Strong offer" ? "buy" : a.label === "Long shot" ? "sell" : "hold";
  const r = a.league_accept_rate;
  return (
    <div className="odds-card">
      <div className="row" style={{ gap: 8 }}>
        <b>Would {teamName} accept?</b>
        <span className={`badge ${cls}`}>{a.label}</span>
      </div>
      <ul className="small" style={{ margin: "6px 0 0", paddingLeft: 18 }}>
        {a.reasons.map((x, i) => <li key={i}>{x}</li>)}
      </ul>
      <p className="small muted" style={{ marginTop: 6 }}>
        Estimated chance: about {Math.round(a.probability * 100)}%. In your league, {r.completed} of {r.proposals} trade
        offers this season were accepted, so even good offers often get turned down. This is an estimate built from what
        usually drives acceptance, not a trained prediction.
      </p>
    </div>
  );
}
