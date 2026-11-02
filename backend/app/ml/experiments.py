"""Experiments: test each accuracy idea against the same walk-forward backtest
and keep only what helps.

    1. Original model (baseline)
    2-6. Original + each new input group on its own
    7. All groups that helped, together
    8. + weighting recent seasons more
    9. Tuned model settings (tuned on 2023, a season the final test never uses)
   10. Retraining every 4 weeks during the season (how the app is really used)
   then: stats/direct ensemble + per-position Sleeper blend, and the injury table.

Run from backend/:  ..\\.venv\\Scripts\\python -m app.ml.experiments
Results go to data/models/experiments.json and are shown in Model lab.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone

from ..data.store import DataStore, get_store
from . import dataset as D
from .model import (MODEL_DIR, Config, backtest_predictions, baselines, eval_frame, improves,
                    metrics, train_and_save)

CANDIDATE_GROUPS = ["sleeper", "depth", "qb", "defense", "weather"]
PARAM_GRID = [
    {"name": "current", "params": {}, "rounds": 400},
    {"name": "smaller trees, more of them", "params": {"num_leaves": 15, "learning_rate": 0.03}, "rounds": 700},
    {"name": "larger trees", "params": {"num_leaves": 63, "min_data_in_leaf": 100}, "rounds": 350},
    {"name": "stronger regularization", "params": {"lambda_l2": 10.0, "min_data_in_leaf": 150, "feature_fraction": 0.6}, "rounds": 500},
    {"name": "slow and steady", "params": {"learning_rate": 0.02, "num_leaves": 31, "feature_fraction": 0.7}, "rounds": 900},
]


def run(store: DataStore | None = None, log=print) -> dict:
    t0 = time.time()
    store = store or get_store()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    log("Loading history and building every candidate input")
    hist = D.load_history(store.season, log)
    ds = D.build(store, hist, log=log)
    seasons = [s for s in (store.season - 2, store.season - 1) if (ds["season"] == s).any()]
    base = baselines(hist, seasons, log)
    results: list[dict] = []

    def trial(cfg: Config, label: str, description: str, retrain_every=None) -> dict:
        ev = eval_frame(backtest_predictions(ds, cfg, seasons, retrain_every, log=log), base)
        m = metrics(ev, "ml_stats")
        r = {"name": cfg.name, "label": label, "description": description, "config": cfg.to_dict(),
             "retrain_every": retrain_every, "metrics": m, "games": int(len(ev))}
        log(f"  => {label}: avg miss {m['all']['mae']:.3f}, ranking {m['all']['rank_corr']:.4f}")
        results.append(r)
        return r

    # 1. baseline
    best_cfg = Config(name="original")
    best = trial(best_cfg, "Original model", "The model as first built: 71 inputs, default settings.")

    # 2-6. each group alone
    helped = []
    for g in CANDIDATE_GROUPS:
        cfg = Config(name=f"+{g}", groups=["base", g])
        r = trial(cfg, f"+ {D.GROUP_LABELS[g]}", f"Original inputs plus: {D.GROUP_LABELS[g]}.")
        r["kept"] = improves(r["metrics"], best["metrics"])
        if r["kept"]:
            helped.append(g)
    best["kept"] = None  # the baseline everything is compared against

    # 7. combine everything that helped on its own
    if helped:
        cfg = Config(name="combined", groups=["base", *helped])
        r = trial(cfg, "All helpful inputs together", "Every input group that helped on its own: "
                  + "; ".join(D.GROUP_LABELS[g] for g in helped) + ".")
        r["kept"] = improves(r["metrics"], best["metrics"])
        if r["kept"]:
            best, best_cfg = r, cfg
        else:
            # Fall back to the single best group if the combination doesn't stack.
            singles = [x for x in results if x.get("kept") and x["name"].startswith("+")]
            if singles:
                top = min(singles, key=lambda x: x["metrics"]["all"]["mae"])
                best, best_cfg = top, Config.from_dict(top["config"])

    # 8. recency weighting
    cfg = Config(**{**best_cfg.to_dict(), "name": "recency", "season_decay": 0.85})
    r = trial(cfg, "Weight recent seasons more", "Each season back counts 15% less, since the game keeps changing.")
    r["kept"] = improves(r["metrics"], best["metrics"])
    if r["kept"]:
        best, best_cfg = r, cfg

    # 9. tuning, judged on 2023 so the 2024-25 test stays untouched
    log("Tuning model settings on 2023 (not used in the final test)")
    tune_season = min(seasons) - 1
    tune = []
    for p in PARAM_GRID:
        cfg = Config(**{**best_cfg.to_dict(), "name": f"tune: {p['name']}", "params": {**best_cfg.params, **p["params"]},
                        "rounds": p["rounds"]})
        ev = eval_frame(backtest_predictions(ds, cfg, [tune_season], log=lambda *_: None),
                        baselines(hist, [tune_season], log=lambda *_: None))
        m = metrics(ev, "ml_stats")
        tune.append({"name": p["name"], "params": p["params"], "rounds": p["rounds"], "metrics_2023": m["all"]})
        log(f"  2023 with {p['name']}: avg miss {m['all']['mae']:.3f}, ranking {m['all']['rank_corr']:.4f}")
    winner = min(tune, key=lambda x: (x["metrics_2023"]["mae"], -x["metrics_2023"]["rank_corr"]))
    if winner["name"] != "current":
        cfg = Config(**{**best_cfg.to_dict(), "name": "tuned", "params": {**best_cfg.params, **winner["params"]},
                        "rounds": winner["rounds"]})
        r = trial(cfg, f"Tuned settings ({winner['name']})", "Model settings picked on 2023, then tested on 2024-25.")
        r["kept"] = improves(r["metrics"], best["metrics"])
        if r["kept"]:
            best, best_cfg = r, cfg

    # 10. in-season retraining (informational: the app retrains weekly anyway)
    r = trial(Config(**{**best_cfg.to_dict(), "name": "retrain4"}), "Retrain every 4 weeks in-season",
              "Same model, but retrained during each test season as new games come in, like the app does weekly.",
              retrain_every=4)
    r["kept"] = None
    retrain_gain = {"mae": round(best["metrics"]["all"]["mae"] - r["metrics"]["all"]["mae"], 3),
                    "rank_corr": round(r["metrics"]["all"]["rank_corr"] - best["metrics"]["all"]["rank_corr"], 4)}

    best_cfg.name = "chosen"
    out = {
        "ran_at": datetime.now(timezone.utc).isoformat(),
        "seasons": seasons,
        "tune_season": tune_season,
        "rule": "A change is kept if it lowers average miss by 0.003+ without hurting ranking, "
                "or raises ranking accuracy by 0.003+ without adding error.",
        "experiments": results,
        "tuning": tune,
        "retrain_gain": retrain_gain,
        "chosen": best_cfg.to_dict(),
        "chosen_label": ", ".join(D.GROUP_LABELS[g] for g in best_cfg.groups),
        "seconds": round(time.time() - t0, 1),
    }
    (MODEL_DIR / "experiments.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    log(f"Experiments done in {out['seconds'] / 60:.0f} min. Chosen: {out['chosen_label']}")

    log("Training the final model with the chosen settings")
    meta = train_and_save(store, best_cfg, log, prepared=(hist, ds))
    # Before/after summary on the same games
    orig = results[0]["metrics"]["all"]
    out["final"] = {"original": orig, "chosen_model": best["metrics"]["all"],
                    "blend": next(m for m in meta["backtest"]["methods"] if m["key"] == "blend")["all"],
                    "sleeper": next(m for m in meta["backtest"]["methods"] if m["key"] == "sleeper_ppr")["all"]}
    (MODEL_DIR / "experiments.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def load() -> dict | None:
    f = MODEL_DIR / "experiments.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else None


if __name__ == "__main__":
    run()
