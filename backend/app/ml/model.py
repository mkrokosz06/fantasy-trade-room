"""Train, backtest and run the LightGBM projection models.

Models (one set per position group: QB, RB, REC = WR+TE):
  * stat models -- each counting stat (targets, carries, yards, TDs, ...),
    so projections work with any scoring system
  * ppr        -- direct PPR points model
  * ppr_q10/q90 -- 10th / 90th percentile PPR points (the floor / ceiling range)

All are conditional on the player playing a full game; availability (injury
status) is applied afterwards, same as the v1 engine.

Backtest is walk-forward: to test season S, train only on seasons before S,
then predict every week of S using pre-game features.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from types import SimpleNamespace

import lightgbm as lgb
import numpy as np
import pandas as pd

from ..config import DATA_DIR
from ..data import sources
from ..data.store import DataStore
from ..projections import build_baseline
from ..scoring import STATS, score
from . import dataset as D

MODEL_DIR = DATA_DIR / "models"
GROUPS = {"QB": ["QB"], "RB": ["RB"], "REC": ["WR", "TE"]}
STAT_TARGETS = {
    "QB": [("pass_att", "poisson"), ("pass_cmp", "poisson"), ("pass_yd", "regression"),
           ("pass_td", "poisson"), ("pass_int", "poisson"), ("rush_att", "poisson"),
           ("rush_yd", "regression"), ("rush_td", "poisson")],
    "RB": [("rush_att", "poisson"), ("rush_yd", "regression"), ("rush_td", "poisson"),
           ("rec_tgt", "poisson"), ("rec", "poisson"), ("rec_yd", "regression"), ("rec_td", "poisson")],
    "REC": [("rec_tgt", "poisson"), ("rec", "poisson"), ("rec_yd", "regression"), ("rec_td", "poisson"),
            ("rush_att", "poisson"), ("rush_yd", "regression"), ("rush_td", "poisson")],
}
POINT_TARGETS = [("ppr", "regression", None), ("ppr_q10", "quantile", 0.1), ("ppr_q90", "quantile", 0.9)]
PARAMS = {"learning_rate": 0.04, "num_leaves": 31, "min_data_in_leaf": 60, "feature_fraction": 0.8,
          "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 2.0, "verbose": -1, "seed": 7}
ROUNDS = 400


# ---------------------------------------------------------------- core fit/predict
def _fit_one(X: pd.DataFrame, y: pd.Series, objective: str, alpha: float | None) -> lgb.Booster:
    params = {**PARAMS, "objective": objective}
    if alpha is not None:
        params["alpha"] = alpha
    if objective == "poisson":
        y = y.clip(lower=0)
    ds = lgb.Dataset(X, label=y, free_raw_data=True)
    return lgb.train(params, ds, num_boost_round=ROUNDS)


def fit_all(train: pd.DataFrame, log=print) -> dict[str, dict[str, lgb.Booster]]:
    models: dict[str, dict[str, lgb.Booster]] = {}
    for g, positions in GROUPS.items():
        part = train[train["position"].isin(positions)]
        X = part[D.FEATURES]
        models[g] = {}
        for stat, obj in STAT_TARGETS[g]:
            models[g][stat] = _fit_one(X, part[stat], obj, None)
        for name, obj, alpha in POINT_TARGETS:
            models[g][name] = _fit_one(X, part["ppr"], obj, alpha)
        log(f"    {g}: {len(models[g])} models on {len(part):,} games")
    return models


def predict_all(models, rows: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=rows.index)
    for g, positions in GROUPS.items():
        m = rows["position"].isin(positions)
        if not m.any():
            continue
        X = rows.loc[m, D.FEATURES]
        for name, booster in models[g].items():
            out.loc[m, name] = booster.predict(X)
    # Keep quantiles ordered around the median.
    if {"ppr", "ppr_q10", "ppr_q90"} <= set(out.columns):
        out["ppr_q10"] = np.minimum(out["ppr_q10"], out["ppr"])
        out["ppr_q90"] = np.maximum(out["ppr_q90"], out["ppr"])
    return out


def stats_to_ppr(pred: pd.DataFrame, positions: pd.Series) -> pd.Series:
    st = pd.DataFrame(0.0, index=pred.index, columns=STATS)
    for c in STATS:
        if c in pred.columns:
            st[c] = pred[c].fillna(0.0)
    return score(st, D.PPR, positions)


# ---------------------------------------------------------------- backtest
def _v1_core(store: DataStore, hist: pd.DataFrame, season: int, weeks: list[int]) -> pd.DataFrame:
    """The v1 engine's usage x regressed-efficiency core, rebuilt point-in-time."""
    players = pd.DataFrame({"position": hist.groupby("id")["position"].last()})
    out = []
    for wk in weeks:
        h = hist[((hist["season"] < season) | ((hist["season"] == season) & (hist["week"] < wk)))
                 & (hist["season"] >= season - 2)]
        fake = SimpleNamespace(weekly=h, players=players, season=season)
        base, _, _ = build_baseline(fake, 5.0)
        pts = score(base.reindex(columns=STATS).fillna(0), D.PPR, base["position"])
        out.append(pd.DataFrame({"id": base.index, "season": season, "week": wk, "v1": pts.values}))
    return pd.concat(out, ignore_index=True)


def _sleeper_hist(season: int, weeks: list[int], log) -> pd.DataFrame:
    cw = sources.id_crosswalk()[["sleeper_id", "gsis_id"]].dropna().drop_duplicates("sleeper_id")
    m = dict(zip(cw["sleeper_id"], cw["gsis_id"]))
    frames = []
    for wk in weeks:
        try:
            f = sources.sleeper_projections_ppr(season, wk)
        except Exception as e:  # noqa: BLE001
            log(f"    no Sleeper projections for {season} week {wk} ({e.__class__.__name__})")
            continue
        if len(f):
            f["id"] = f["sleeper_id"].map(m)
            frames.append(f.dropna(subset=["id"]).drop_duplicates(["id", "week"]))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["id", "season", "week", "sleeper_ppr"])


def _metrics(ev: pd.DataFrame, col: str) -> dict:
    err = ev[col] - ev["actual"]
    rho = []
    for _, g in ev.groupby(["season", "week", "position"]):
        if len(g) >= 8:
            rho.append(g[col].rank().corr(g["actual"].rank()))
    return {"mae": round(float(err.abs().mean()), 3), "bias": round(float(err.mean()), 3),
            "rank_corr": round(float(np.nanmean(rho)), 4), "n": int(len(ev))}


def backtest(store: DataStore, ds: pd.DataFrame, hist: pd.DataFrame, log=print) -> dict:
    seasons = [s for s in (store.season - 2, store.season - 1)
               if s >= D.FIRST_SEASON + 3 and (ds["season"] == s).any()]
    labeled = ds[~ds["is_future"] & ~ds["partial"]]
    evals = []
    for S in seasons:
        log(f"  Backtest {S}: training on {D.FIRST_SEASON}-{S - 1}")
        models = fit_all(labeled[labeled["season"] < S], log)
        test = labeled[labeled["season"] == S]
        pred = predict_all(models, test)
        weeks = sorted(test["week"].unique().tolist())
        log(f"  Backtest {S}: rebuilding v1 week by week and loading Sleeper's projections")
        v1 = _v1_core(store, hist, S, weeks)
        sl = _sleeper_hist(S, weeks, log)
        ev = test[["id", "name", "position", "season", "week", "ppr", "ppr_e3"]].rename(
            columns={"ppr": "actual", "ppr_e3": "rolling"})
        ev["ml_stats"] = stats_to_ppr(pred, test["position"]).values
        ev["ml_direct"] = pred["ppr"].values
        ev["q10"] = pred["ppr_q10"].values
        ev["q90"] = pred["ppr_q90"].values
        ev = ev.merge(v1, on=["id", "season", "week"], how="left")
        ev = ev.merge(sl[["id", "season", "week", "sleeper_ppr"]], on=["id", "season", "week"], how="left")
        evals.append(ev)
    ev = pd.concat(evals, ignore_index=True)
    # Compare on the same players: fantasy-relevant games where every method has a number.
    ev = ev.dropna(subset=["rolling", "v1", "sleeper_ppr", "ml_stats"])
    ev = ev[ev["sleeper_ppr"] >= 3].copy()

    # Best blend of the model with Sleeper (what the app will use).
    best_w, best_mae = 0.0, np.inf
    for w in np.round(np.arange(0, 1.01, 0.05), 2):
        mae = float(((1 - w) * ev["ml_stats"] + w * ev["sleeper_ppr"] - ev["actual"]).abs().mean())
        if mae < best_mae:
            best_w, best_mae = float(w), mae
    ev["blend"] = (1 - best_w) * ev["ml_stats"] + best_w * ev["sleeper_ppr"]

    methods = {"sleeper_ppr": "Sleeper projections", "v1": "Current engine (v1 core)",
               "rolling": "Rolling average (last ~3 games)", "ml_stats": "Model: stats, scored PPR",
               "ml_direct": "Model: direct PPR", "blend": f"Model + Sleeper blend ({int((1 - best_w) * 100)}/{int(best_w * 100)})"}
    report = {"seasons": seasons, "games": int(len(ev)), "blend_sleeper_weight": best_w, "methods": []}
    for col, label in methods.items():
        row = {"key": col, "label": label, "all": _metrics(ev, col)}
        for p in D.POS:
            row[p] = _metrics(ev[ev["position"] == p], col)
        report["methods"].append(row)
    # Floor/ceiling should contain 80% of outcomes. Quantile models usually come
    # out a bit narrow, so find the widening factor that hits 80% in the backtest.
    def cover(k: float) -> float:
        lo = ev["ml_direct"] - k * (ev["ml_direct"] - ev["q10"])
        hi = ev["ml_direct"] + k * (ev["q90"] - ev["ml_direct"])
        return float(((ev["actual"] >= lo) & (ev["actual"] <= hi)).mean())

    raw = cover(1.0)
    scale = next((float(k) for k in np.round(np.arange(1.0, 2.51, 0.02), 2) if cover(k) >= 0.8), 2.5)
    report["range_coverage_raw"] = round(raw, 3)
    report["range_scale"] = scale
    report["range_coverage"] = round(cover(scale), 3)
    width = scale * (ev["q90"] - ev["q10"])
    report["range_width"] = round(float(width.mean()), 2)
    return report


# ---------------------------------------------------------------- full pipeline
def train_and_save(store: DataStore, log=print) -> dict:
    t0 = time.time()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    log("Loading history (first run downloads ~45 small files)")
    hist = D.load_history(store.season, log)
    log("Building pre-game features")
    ds = D.build(store, hist)
    log(f"  {int((~ds['is_future']).sum()):,} historical games, {int(ds['is_future'].sum()):,} upcoming")

    report = backtest(store, ds, hist, log)

    log("Training final models on every season through this week")
    labeled = ds[~ds["is_future"] & ~ds["partial"]]
    models = fit_all(labeled, log)
    for g, ms in models.items():
        for name, b in ms.items():
            b.save_model(str(MODEL_DIR / f"{g}_{name}.txt"))

    importance = {}
    for g, ms in models.items():
        imp = pd.Series(ms["ppr"].feature_importance("gain"), index=D.FEATURES)
        imp = (imp / imp.sum()).sort_values(ascending=False).head(12)
        importance[g] = [{"feature": f, "label": D.label_feature(f), "share": round(float(v), 4)}
                         for f, v in imp.items()]
    meta = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "seasons": [D.FIRST_SEASON, store.season],
        "through_week": store.last_stats_week,
        "training_games": int(len(labeled)),
        "features": D.FEATURES,
        "n_features": len(D.FEATURES),
        "backtest": report,
        "importance": importance,
        "seconds": round(time.time() - t0, 1),
    }
    (MODEL_DIR / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    log(f"Done in {meta['seconds']}s")
    return meta


def load_meta() -> dict | None:
    f = MODEL_DIR / "meta.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else None


def load_models() -> dict[str, dict[str, lgb.Booster]]:
    models: dict[str, dict[str, lgb.Booster]] = {}
    for g in GROUPS:
        models[g] = {}
        for stat, _ in STAT_TARGETS[g]:
            models[g][stat] = lgb.Booster(model_file=str(MODEL_DIR / f"{g}_{stat}.txt"))
        for name, _, _ in POINT_TARGETS:
            models[g][name] = lgb.Booster(model_file=str(MODEL_DIR / f"{g}_{name}.txt"))
    return models


def predict_future(store: DataStore, end_week: int = 18, range_scale: float = 1.0) -> pd.DataFrame:
    """Per-game model predictions for every upcoming game (id, week, stats, ppr range)."""
    hist = D.load_history(store.season, log=lambda *_: None)
    ds = D.build(store, hist, end_week)
    fut = ds[ds["is_future"]]
    pred = predict_all(load_models(), fut)
    pred["ppr_q10"] = (pred["ppr"] - range_scale * (pred["ppr"] - pred["ppr_q10"])).clip(lower=0)
    pred["ppr_q90"] = pred["ppr"] + range_scale * (pred["ppr_q90"] - pred["ppr"])
    out = pd.concat([fut[["id", "week", "position"]], pred], axis=1)
    out = out.rename(columns={"ppr": "ppr_model"})
    return out.reset_index(drop=True)
