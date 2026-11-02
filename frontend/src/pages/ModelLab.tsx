import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api, invalidateAll, useApi, type MetricRow, type MlStatus } from "../api";
import { ErrorBox, Loading } from "../components/bits";

const SCOPES = ["all", "QB", "RB", "WR", "TE"] as const;
const GROUP_LABEL: Record<string, string> = { QB: "Quarterbacks", RB: "Running backs", REC: "Receivers and tight ends" };

export default function ModelLab() {
  const st = useApi<MlStatus>("/api/ml/status");
  const [scope, setScope] = useState<(typeof SCOPES)[number]>("all");
  const [err, setErr] = useState<string | null>(null);
  const training = st.data?.status === "training";
  const { reload } = st;

  // Poll while training; refresh every model view once it finishes.
  useEffect(() => {
    if (!training) return;
    const t = setInterval(reload, 2500);
    return () => clearInterval(t);
  }, [training, reload]);
  const [wasTraining, setWasTraining] = useState(false);
  useEffect(() => {
    if (training) setWasTraining(true);
    else if (wasTraining) {
      setWasTraining(false);
      invalidateAll();
    }
  }, [training, wasTraining]);

  const train = async () => {
    setErr(null);
    try {
      await api("/api/ml/train", { method: "POST" });
      reload();
    } catch (e) {
      setErr((e as Error).message);
    }
  };

  if (!st.data) return st.error ? <ErrorBox error={st.error} /> : <Loading what="Loading model status" />;
  const meta = st.data.meta;
  const bt = meta?.backtest;
  const rows = bt?.methods ?? [];
  const metric = (r: (typeof rows)[number]) => r[scope] as MetricRow;
  const bestMae = rows.length ? Math.min(...rows.map((r) => metric(r).mae)) : 0;
  const bestRank = rows.length ? Math.max(...rows.map((r) => metric(r).rank_corr)) : 0;

  return (
    <div className="stack">
      <div className="page-head">
        <div>
          <h1>Model lab</h1>
          <p>
            A machine-learning model trained on every regular-season game since 2016. The tools in this section run on
            it; the main tools keep the original engine, so you can compare the two side by side.
          </p>
        </div>
        <button className="btn" onClick={train} disabled={training}>
          {training ? "Training…" : meta ? "Retrain model" : "Train model"}
        </button>
      </div>
      <ErrorBox error={err || st.data.error} />

      <section className="panel stack">
        {meta ? (
          <div className="row" style={{ gap: 32 }}>
            <div className="stat"><b>{new Date(meta.trained_at).toLocaleDateString()}</b><span>Trained</span></div>
            <div className="stat"><b>Week {meta.through_week}</b><span>Data through</span></div>
            <div className="stat"><b>{meta.training_games.toLocaleString()}</b><span>Games learned from</span></div>
            <div className="stat"><b>{meta.n_features}</b><span>Inputs per prediction</span></div>
          </div>
        ) : (
          <p>
            The model hasn't been trained yet. Training downloads about 45 small data files the first time, tests the
            model on the 2024 and 2025 seasons, then trains the final version. It takes 5 to 8 minutes.
          </p>
        )}
        <p className="small muted">
          Retrain once a week after games finish, so the model learns from the latest results. The Model lab tools
          use it right away.
        </p>
        {(training || st.data.log.length > 0) && <div className="log" aria-live="polite">{st.data.log.join("\n") || "Starting…"}</div>}
      </section>

      {bt && (
        <section className="stack">
          <div className="side-head">
            <h2>How it performed on seasons it never saw</h2>
            <div className="seg" role="group" aria-label="Position">
              {SCOPES.map((s) => (
                <button key={s} aria-pressed={scope === s} onClick={() => setScope(s)}>
                  {s === "all" ? "All" : s}
                </button>
              ))}
            </div>
          </div>
          <p className="muted">
            Trained only on earlier seasons, then predicted every week of {bt.seasons.join(" and ")} using only what was known
            before kickoff: {bt.games.toLocaleString()} player-games, same players for every method.
            <b> Average miss</b> is how far off a projection is, in points (lower is better).
            <b> Ranking accuracy</b> measures how well it orders players at a position each week, from 0 to 1 (higher is
            better). That's what start/sit and trade decisions depend on.
          </p>
          <div className="tablewrap">
            <table>
              <thead>
                <tr>
                  <th>Method</th>
                  <th className="n">Average miss</th>
                  <th className="n">Ranking accuracy</th>
                  <th className="n" title="Average of projected minus actual. Positive means it projects too high.">Bias</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((r) => {
                  const m = metric(r);
                  return (
                    <tr key={r.key}>
                      <td>{r.label}{r.key === "blend" && <span className="badge hold" style={{ marginLeft: 8 }}>Used in Model lab</span>}</td>
                      <td className={`n ${m.mae === bestMae ? "best" : ""}`}>{m.mae.toFixed(2)}</td>
                      <td className={`n ${m.rank_corr === bestRank ? "best" : ""}`}>{m.rank_corr.toFixed(3)}</td>
                      <td className="n muted">{m.bias > 0 ? "+" : ""}{m.bias.toFixed(2)}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
          <p className="small muted">
            Green marks the best number in each column. Fantasy points are noisy: even the best projections miss by about 5
            points per player per game, so small differences here are real gains.
          </p>
        </section>
      )}

      {bt && (
        <div className="grid2">
          <section className="panel stack">
            <h2>Floor and ceiling</h2>
            <p>
              Besides a projection, the model gives each player a range for every game. The <b>floor</b> is a bad-but-realistic
              week: they score less than this only 1 week in 10. The <b>ceiling</b> is a great week: they beat it only 1 week in 10.
              So 8 weeks out of 10 land between the two.
            </p>
            <p className="small muted">
              Two players can both project 14 points, one with a range of 10–18 and the other 4–28. The first is the safer
              start; the second is the one who can win you a week. In testing, {Math.round(bt.range_coverage * 100)}% of actual
              results landed inside the range{bt.range_coverage_raw !== undefined && bt.range_scale && bt.range_scale > 1
                ? ` (after widening it ${Math.round((bt.range_scale - 1) * 100)}%, because the raw model was a little overconfident at ${Math.round(bt.range_coverage_raw * 100)}%)`
                : ""}.
            </p>
          </section>
          <section className="panel stack">
            <h2>Use it</h2>
            <p className="muted">These run on the model blended with Sleeper, adjusted for injuries.</p>
            <div className="row">
              <Link className="btn" to="/model/trade">Trade calculator</Link>
              <Link className="btn ghost" to="/model/advisor">Trade advisor</Link>
              <Link className="btn ghost" to="/model/rankings">Rankings</Link>
            </div>
          </section>
        </div>
      )}

      {meta && (
        <section className="stack">
          <h2>What the model pays attention to</h2>
          <div className="grid2">
            {Object.entries(meta.importance).map(([g, feats]) => (
              <section key={g} className="panel">
                <h3>{GROUP_LABEL[g] ?? g}</h3>
                <div className="bars">
                  {feats.slice(0, 8).map((f) => (
                    <div key={f.feature} className="bar-row">
                      <span>{f.label}</span>
                      <i style={{ width: `${(f.share / feats[0].share) * 100}%` }} />
                      <span className="num small muted">{(f.share * 100).toFixed(0)}%</span>
                    </div>
                  ))}
                </div>
              </section>
            ))}
          </div>
        </section>
      )}
    </div>
  );
}
