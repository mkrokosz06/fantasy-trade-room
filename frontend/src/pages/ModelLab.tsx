import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api, invalidateAll, useApi, type Experiments, type InjuryRow, type MetricRow, type MlStatus } from "../api";
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

  const train = async (kind: "train" | "experiments" = "train") => {
    setErr(null);
    try {
      await api(`/api/ml/train?kind=${kind}`, { method: "POST" });
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
        <div className="row">
          <button
            className="btn ghost"
            onClick={() => train("experiments")}
            disabled={training}
            title="Tests every accuracy idea against past seasons, keeps what helps, then retrains. About 70 minutes."
          >
            Re-run experiments
          </button>
          <button className="btn" onClick={() => train("train")} disabled={training}>
            {training ? (st.data.kind === "experiments" ? "Running experiments…" : "Training…") : meta ? "Retrain model" : "Train model"}
          </button>
        </div>
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

      {st.data.experiments && <ExperimentsSection ex={st.data.experiments} />}

      {meta?.injury?.rows && <InjurySection rows={meta.injury.rows} />}

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

      {meta?.backtest.blend_by_pos && (
        <p className="small muted">
          How much each position leans on Sleeper in the blend:{" "}
          {Object.entries(meta.backtest.blend_by_pos)
            .map(([p, w]) => `${p} ${Math.round(w * 100)}%`)
            .join(", ")}
          . Picked by the backtest; the rest comes from the model.
        </p>
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

function delta(v: number, d = 3, invert = false) {
  if (Math.abs(v) < 10 ** -d / 2) return <span className="muted">±0</span>;
  const good = invert ? v < 0 : v > 0;
  return (
    <span className={`delta ${good ? "up" : "down"}`}>
      {v > 0 ? "+" : ""}
      {v.toFixed(d)}
    </span>
  );
}

function ExperimentsSection({ ex }: { ex: Experiments }) {
  const base = ex.experiments[0]?.metrics.all;
  return (
    <section className="stack">
      <h2>What we tried to make it more accurate</h2>
      <p className="muted">
        Each idea was tested on the same {ex.seasons.join(" and ")} games, trained only on earlier data. {ex.rule}{" "}
        Model settings were tuned on {ex.tune_season}, so the final test seasons stayed untouched.
      </p>
      <div className="tablewrap">
        <table>
          <thead>
            <tr>
              <th>Idea</th>
              <th className="n">Average miss</th>
              <th className="n" title="Change vs the original model. Negative is better.">vs original</th>
              <th className="n">Ranking accuracy</th>
              <th className="n" title="Change vs the original model. Positive is better.">vs original</th>
              <th>Result</th>
            </tr>
          </thead>
          <tbody>
            {ex.experiments.map((e) => {
              const m = e.metrics.all;
              return (
                <tr key={e.name}>
                  <td style={{ whiteSpace: "normal", minWidth: 260 }}>
                    <b>{e.label}</b>
                    <div className="small muted">{e.description}</div>
                  </td>
                  <td className="n">{m.mae.toFixed(3)}</td>
                  <td className="n">{base ? delta(m.mae - base.mae, 3, true) : "–"}</td>
                  <td className="n">{m.rank_corr.toFixed(3)}</td>
                  <td className="n">{base ? delta(m.rank_corr - base.rank_corr, 3) : "–"}</td>
                  <td>
                    {e.kept === null || e.kept === undefined ? (
                      <span className="badge hold">{e.retrain_every ? "Measured" : "Baseline"}</span>
                    ) : e.kept ? (
                      <span className="badge buy">Kept</span>
                    ) : (
                      <span className="badge sell">Didn't help</span>
                    )}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <p className="small muted">
        Final model uses: {ex.chosen_label}. Retraining during the season (as the app does weekly) changed average miss by{" "}
        {(-ex.retrain_gain.mae).toFixed(3)} and ranking by {ex.retrain_gain.rank_corr >= 0 ? "+" : ""}
        {ex.retrain_gain.rank_corr.toFixed(3)}, so real-world accuracy should be at least what's shown here.
      </p>
      <details className="panel">
        <summary>Settings tried on {ex.tune_season}</summary>
        <table style={{ marginTop: 8 }}>
          <thead>
            <tr><th>Settings</th><th className="n">Average miss</th><th className="n">Ranking accuracy</th></tr>
          </thead>
          <tbody>
            {ex.tuning.map((t) => (
              <tr key={t.name}>
                <td>{t.name}</td>
                <td className="n">{t.metrics_2023.mae.toFixed(3)}</td>
                <td className="n">{t.metrics_2023.rank_corr.toFixed(3)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </details>
    </section>
  );
}

const PRACTICE_LABEL: Record<string, string> = {
  DNP: "Missed practice", Limited: "Limited practice", Full: "Full practice", None: "No practice report",
};

function InjurySection({ rows }: { rows: InjuryRow[] }) {
  const shown = rows.filter((r) => r.n >= 25 && r.status !== "Out");
  return (
    <section className="stack">
      <h2>How injury designations really play out</h2>
      <p className="muted">
        Learned from every injury report since 2016: how much of a player's usual output they actually produced,
        counting games they sat out or left early. Model lab uses these for each player's next game, based on this
        week's official report when it's out (Wednesday to Friday), or Sleeper's status before that.
      </p>
      <div className="tablewrap">
        <table>
          <thead>
            <tr>
              <th>Game status</th>
              <th>Practice that week</th>
              <th className="n">Times seen</th>
              <th className="n">Played</th>
              <th className="n" title="Share of the player's usual fantasy output">Expected output</th>
            </tr>
          </thead>
          <tbody>
            {shown.map((r) => (
              <tr key={`${r.status}-${r.practice}`}>
                <td>{r.status === "None" ? "No game status (listed on report)" : r.status}</td>
                <td>{PRACTICE_LABEL[r.practice] ?? r.practice}</td>
                <td className="n">{r.n.toLocaleString()}</td>
                <td className="n">{Math.round(r.played * 100)}%</td>
                <td className="n"><b>{Math.round(r.multiplier * 100)}%</b></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="small muted">Players listed Out essentially never play, so they're projected at zero for that game.</p>
    </section>
  );
}
