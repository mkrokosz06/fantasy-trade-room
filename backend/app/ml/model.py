"""Train, backtest and run the LightGBM projection models.

Models (one set per position group: QB, RB, REC = WR+TE):
  * stat models -- each counting stat (targets, carries, yards, TDs, ...),
    so projections work with any scoring system
  * ppr        -- direct PPR points model
  * ppr_q10/q90 -- 10th / 90th percentile PPR points (the floor / ceiling range)

All are conditional on the player playing a full game; availability (injury
status) is applied afterwards.

Backtest is walk-forward: to test season S, train only on data before S (or,
with `retrain_every`, before each block of weeks in S), then predict using
pre-game features. Which features and settings to use is decided by the
experiments in experiments.py and saved as the "chosen config".
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
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
POS_GROUP = {"QB": "QB", "RB": "RB", "WR": "REC", "TE": "REC"}
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
BASE_PARAMS = {"learning_rate": 0.04, "num_leaves": 31, "min_data_in_leaf": 60, "feature_fraction": 0.8,
               "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 2.0, "verbose": -1, "seed": 7}


@dataclass
class Config:
    """Everything that defines a model variant."""
    name: str = "original"
    groups: list[str] = field(default_factory=lambda: ["base"])
    params: dict = field(default_factory=dict)       # overrides of BASE_PARAMS
    rounds: int = 400
    season_decay: float = 1.0                         # sample weight multiplier per season back

    @property
    def features(self) -> list[str]:
        return D.features_for(self.groups)

    @property
    def lgb_params(self) -> dict:
        return {**BASE_PARAMS, **self.params}

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------- fit / predict
def _fit_one(X, y, w, cfg: Config, objective: str, alpha: float | None) -> lgb.Booster:
    params = {**cfg.lgb_params, "objective": objective}
    if alpha is not None:
        params["alpha"] = alpha
    if objective == "poisson":
        y = y.clip(lower=0)
    return lgb.train(params, lgb.Dataset(X, label=y, weight=w, free_raw_data=True),
                     num_boost_round=cfg.rounds)


def fit_all(train: pd.DataFrame, cfg: Config, quantiles: bool = True, log=print) -> dict:
    feats = cfg.features
    models: dict[str, dict[str, lgb.Booster]] = {}
    for g, positions in GROUPS.items():
        part = train[train["position"].isin(positions)]
        X = part[feats]
        w = None
        if cfg.season_decay < 1:
            w = (cfg.season_decay ** (part["season"].max() - part["season"])).to_numpy()
        models[g] = {stat: _fit_one(X, part[stat], w, cfg, obj, None) for stat, obj in STAT_TARGETS[g]}
        for name, obj, alpha in POINT_TARGETS:
            if quantiles or alpha is None:
                models[g][name] = _fit_one(X, part["ppr"], w, cfg, obj, alpha)
        log(f"    {g}: {len(models[g])} models on {len(part):,} games")
    return models


def predict_all(models: dict, rows: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    out = pd.DataFrame(index=rows.index)
    for g, positions in GROUPS.items():
        m = rows["position"].isin(positions)
        if not m.any():
            continue
        X = rows.loc[m, cfg.features]
        for name, booster in models[g].items():
            out.loc[m, name] = booster.predict(X)
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


# ---------------------------------------------------------------- backtest pieces
def labeled(ds: pd.DataFrame) -> pd.DataFrame:
    return ds[~ds["is_future"] & ~ds["partial"]]


def backtest_predictions(ds: pd.DataFrame, cfg: Config, seasons: list[int], retrain_every: int | None = None,
                         quantiles: bool = False, log=print) -> pd.DataFrame:
    lab = labeled(ds)
    out = []
    for S in seasons:
        test_all = lab[lab["season"] == S]
        blocks = [(1, 99)] if not retrain_every else [(w, w + retrain_every - 1) for w in range(1, 19, retrain_every)]
        for start, end in blocks:
            test = test_all[test_all["week"].between(start, end)]
            if test.empty:
                continue
            train = lab[(lab["season"] < S) | ((lab["season"] == S) & (lab["week"] < start))]
            log(f"  {cfg.name}: {S} weeks {start}-{min(end, 18)}, training on {len(train):,} games")
            models = fit_all(train, cfg, quantiles, log=lambda *_: None)
            pred = predict_all(models, test, cfg)
            ev = test[["id", "name", "position", "season", "week", "ppr", "ppr_e3"]].rename(
                columns={"ppr": "actual", "ppr_e3": "rolling"})
            ev["ml_stats"] = stats_to_ppr(pred, test["position"]).values
            ev["ml_direct"] = pred["ppr"].values
            if quantiles:
                ev["q10"] = pred["ppr_q10"].values
                ev["q90"] = pred["ppr_q90"].values
            out.append(ev)
    return pd.concat(out, ignore_index=True)


def _v1_core(hist: pd.DataFrame, season: int, weeks: list[int]) -> pd.DataFrame:
    """The original engine's usage x regressed-efficiency core, rebuilt point-in-time."""
    players = pd.DataFrame({"position": hist.groupby("id")["position"].last()})
    out = []
    for wk in weeks:
        h = hist[((hist["season"] < season) | ((hist["season"] == season) & (hist["week"] < wk)))
                 & (hist["season"] >= season - 2)]
        base, _, _ = build_baseline(SimpleNamespace(weekly=h, players=players, season=season), 5.0)
        pts = score(base.reindex(columns=STATS).fillna(0), D.PPR, base["position"])
        out.append(pd.DataFrame({"id": base.index, "season": season, "week": wk, "v1": pts.values}))
    return pd.concat(out, ignore_index=True)


def _sleeper_hist(season: int, weeks: list[int]) -> pd.DataFrame:
    cw = sources.id_crosswalk()[["sleeper_id", "gsis_id"]].dropna().drop_duplicates("sleeper_id")
    m = dict(zip(cw["sleeper_id"], cw["gsis_id"]))
    frames = []
    for wk in weeks:
        try:
            f = sources.sleeper_projections_ppr(season, wk)
        except Exception:  # noqa: BLE001
            continue
        if len(f):
            f["id"] = f["sleeper_id"].map(m)
            frames.append(f.dropna(subset=["id"]).drop_duplicates(["id", "week"]))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["id", "season", "week", "sleeper_ppr"])


def baselines(hist: pd.DataFrame, seasons: list[int], log=print) -> pd.DataFrame:
    frames = []
    for S in seasons:
        weeks = sorted(hist.loc[hist["season"] == S, "week"].unique().tolist())
        log(f"  Benchmarks for {S}: original engine week by week, plus Sleeper's projections")
        b = _v1_core(hist, S, weeks).merge(_sleeper_hist(S, weeks)[["id", "season", "week", "sleeper_ppr"]],
                                           on=["id", "season", "week"], how="outer")
        frames.append(b)
    return pd.concat(frames, ignore_index=True)


def eval_frame(ev: pd.DataFrame, base: pd.DataFrame) -> pd.DataFrame:
    """Same fantasy-relevant games for every method."""
    ev = ev.merge(base, on=["id", "season", "week"], how="left")
    ev = ev.dropna(subset=["rolling", "v1", "sleeper_ppr", "ml_stats"])
    return ev[ev["sleeper_ppr"] >= 3].copy()


def metrics(ev: pd.DataFrame, col: str) -> dict:
    def one(e):
        if e.empty:
            return {"mae": None, "bias": None, "rank_corr": None, "n": 0}
        err = e[col] - e["actual"]
        rho = [g[col].rank().corr(g["actual"].rank()) for _, g in e.groupby(["season", "week", "position"]) if len(g) >= 8]
        return {"mae": round(float(err.abs().mean()), 3), "bias": round(float(err.mean()), 3),
                "rank_corr": round(float(np.nanmean(rho)), 4), "n": int(len(e))}
    out = {"all": one(ev)}
    for p in D.POS:
        out[p] = one(ev[ev["position"] == p])
    return out


def improves(new: dict, old: dict) -> bool:
    """Keep a change if it cuts error without hurting ranking, or improves ranking without adding error."""
    n, o = new["all"], old["all"]
    return ((n["mae"] < o["mae"] - 0.003 and n["rank_corr"] >= o["rank_corr"] - 0.002)
            or (n["rank_corr"] > o["rank_corr"] + 0.003 and n["mae"] <= o["mae"] + 0.003))


def posthoc(ev: pd.DataFrame) -> dict:
    """Ensemble weight between stat-based and direct PPR models, then a
    Sleeper blend weight for each position."""
    best_a, best = 1.0, np.inf
    for a in np.round(np.arange(0, 1.01, 0.1), 2):
        mae = float((a * ev["ml_stats"] + (1 - a) * ev["ml_direct"] - ev["actual"]).abs().mean())
        if mae < best:
            best_a, best = float(a), mae
    ev = ev.assign(ml=best_a * ev["ml_stats"] + (1 - best_a) * ev["ml_direct"])

    def best_w(e):
        bw, bm = 0.0, np.inf
        for w in np.round(np.arange(0, 1.01, 0.05), 2):
            m = float(((1 - w) * e["ml"] + w * e["sleeper_ppr"] - e["actual"]).abs().mean())
            if m < bm:
                bw, bm = float(w), m
        return bw

    single = best_w(ev)
    by_pos = {p: best_w(ev[ev["position"] == p]) for p in D.POS}
    ev["blend_single"] = (1 - single) * ev["ml"] + single * ev["sleeper_ppr"]
    w = ev["position"].map(by_pos)
    ev["blend"] = (1 - w) * ev["ml"] + w * ev["sleeper_ppr"]
    return {"ensemble_alpha": best_a, "blend_single": single, "blend_by_pos": by_pos, "frame": ev}


def calibrate_range(ev: pd.DataFrame) -> dict:
    def cover(k):
        lo = ev["ml_direct"] - k * (ev["ml_direct"] - ev["q10"])
        hi = ev["ml_direct"] + k * (ev["q90"] - ev["ml_direct"])
        return float(((ev["actual"] >= lo) & (ev["actual"] <= hi)).mean())
    scale = next((float(k) for k in np.round(np.arange(1.0, 2.51, 0.02), 2) if cover(k) >= 0.8), 2.5)
    return {"range_coverage_raw": round(cover(1.0), 3), "range_scale": scale,
            "range_coverage": round(cover(scale), 3),
            "range_width": round(float((scale * (ev["q90"] - ev["q10"])).mean()), 2)}


# ---------------------------------------------------------------- injury availability
PRACTICE = {"Did Not Participate In Practice": "DNP", "Limited Participation in Practice": "Limited",
            "Full Participation in Practice": "Full"}


def injury_table(ds: pd.DataFrame, current_season: int, log=print) -> dict:
    """Learn how much of a player's usual output to expect given game-status and
    practice participation, from every injury report since 2016.

    multiplier = (actual points / pre-game expectation) for listed players,
    relative to the same ratio for unlisted players. Includes players who
    didn't play (0 points) and those who left early.
    """
    lab = ds[~ds["is_future"]].sort_values("t")
    reps = []
    for s in range(D.FIRST_SEASON, current_season):
        try:
            r = sources.injury_reports(s, current_season)
        except Exception:  # noqa: BLE001
            continue
        r = r[r["game_type"] == "REG"].dropna(subset=["gsis_id", "week"])
        reps.append(pd.DataFrame({"id": r["gsis_id"], "season": int(s), "week": r["week"].astype(int),
                                  "status": r["report_status"], "practice": r["practice_status"].map(PRACTICE)}))
    if not reps:
        return {}
    rep = pd.concat(reps, ignore_index=True).drop_duplicates(["id", "season", "week"])
    rep = rep[rep["id"].isin(lab["id"].unique())]
    rep["t"] = rep["season"] * 100 + rep["week"]
    # Pre-game expectation: the player's rolling 10-game average entering that week.
    exp_src = lab[["id", "t", "ppr", "position", "report_status", "practice_status"]].copy()
    exp_src["exp_after"] = exp_src.groupby("id")["ppr"].transform(lambda s: s.ewm(span=10).mean())
    exp_src["exp_pre"] = exp_src.groupby("id")["exp_after"].shift(1)
    rep["id"] = rep["id"].astype(str)
    exp_src["id"] = exp_src["id"].astype(str)
    rep["t"] = rep["t"].astype("int64")
    exp_src["t"] = exp_src["t"].astype("int64")
    rep = pd.merge_asof(rep.sort_values("t"), exp_src[["id", "t", "exp_after", "position"]].sort_values("t"),
                        on="t", by="id", allow_exact_matches=False, direction="backward")
    act = lab[["id", "season", "week", "ppr"]]
    rep = rep.merge(act, on=["id", "season", "week"], how="left")
    rep["actual"] = rep["ppr"].fillna(0.0)
    rep = rep[rep["position"].isin(D.POS) & (rep["exp_after"] >= 6)]

    unl = exp_src[exp_src["report_status"].isna() & exp_src["practice_status"].isna() & (exp_src["exp_pre"] >= 6)]
    base_ratio = float(unl["ppr"].sum() / unl["exp_pre"].sum())
    rep["status"] = rep["status"].fillna("None")
    rep["practice"] = rep["practice"].fillna("None")
    rep = rep[rep["status"].isin(["None", "Questionable", "Doubtful", "Out"])]

    by_status = rep.groupby("status").apply(lambda g: g["actual"].sum() / g["exp_after"].sum() / base_ratio)
    rows = []
    for (st, pr), g in rep.groupby(["status", "practice"]):
        raw = g["actual"].sum() / g["exp_after"].sum() / base_ratio
        n = len(g)
        k = 40
        shrunk = (n * raw + k * by_status[st]) / (n + k)
        rows.append({"status": st, "practice": pr, "n": int(n), "played": round(float((g["ppr"].notna()).mean()), 3),
                     "multiplier": round(float(min(shrunk, 1.0)), 3)})
    table = {"base_ratio": round(base_ratio, 3),
             "by_status": {k: round(float(min(v, 1.0)), 3) for k, v in by_status.items()},
             "rows": sorted(rows, key=lambda r: (r["status"], r["practice"]))}
    log(f"  Injury table from {len(rep):,} reports: " +
        ", ".join(f"{k} {v:.2f}" for k, v in table["by_status"].items()))
    return table


# ---------------------------------------------------------------- final training
def load_chosen_config() -> Config:
    f = MODEL_DIR / "experiments.json"
    if f.exists():
        try:
            return Config.from_dict(json.loads(f.read_text(encoding="utf-8"))["chosen"])
        except (KeyError, json.JSONDecodeError, TypeError):
            pass
    return Config()


def train_and_save(store: DataStore, cfg: Config | None = None, log=print,
                   prepared: tuple[pd.DataFrame, pd.DataFrame] | None = None) -> dict:
    t0 = time.time()
    cfg = cfg or load_chosen_config()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if prepared is None:
        log("Loading history and building pre-game features")
        hist = D.load_history(store.season, log)
        ds = D.build(store, hist, log=log)
    else:
        hist, ds = prepared
    log(f"Model settings: {cfg.name} ({', '.join(D.GROUP_LABELS[g] for g in cfg.groups)})")

    seasons = [s for s in (store.season - 2, store.season - 1) if (ds["season"] == s).any()]
    ev = backtest_predictions(ds, cfg, seasons, quantiles=True, log=log)
    ev = eval_frame(ev, baselines(hist, seasons, log))
    ph = posthoc(ev)
    ev = ph.pop("frame")
    methods = {"sleeper_ppr": "Sleeper projections", "v1": "Original engine (v1 core)",
               "rolling": "Rolling average (last ~3 games)", "ml_stats": "Model: stats, scored PPR",
               "ml_direct": "Model: direct PPR", "blend": "Model + Sleeper blend"}
    report = {"seasons": seasons, "games": int(len(ev)), "blend_sleeper_weight": ph["blend_single"],
              "blend_by_pos": ph["blend_by_pos"], "ensemble_alpha": ph["ensemble_alpha"],
              "methods": [{"key": k, "label": lbl, **metrics(ev, k)} for k, lbl in methods.items()],
              **calibrate_range(ev)}

    log("Learning injury availability from past injury reports")
    inj = injury_table(ds, store.season, log)

    log("Training final models on every season through this week")
    lab = labeled(ds)
    models = fit_all(lab, cfg, quantiles=True, log=log)
    for g, ms in models.items():
        for name, b in ms.items():
            b.save_model(str(MODEL_DIR / f"{g}_{name}.txt"))
    importance = {}
    for g, ms in models.items():
        imp = pd.Series(ms["ppr"].feature_importance("gain"), index=cfg.features)
        imp = (imp / imp.sum()).sort_values(ascending=False).head(12)
        importance[g] = [{"feature": f, "label": D.label_feature(f), "share": round(float(v), 4)} for f, v in imp.items()]
    meta = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "seasons": [D.FIRST_SEASON, store.season],
        "through_week": store.last_stats_week,
        "training_games": int(len(lab)),
        "config": cfg.to_dict(),
        "features": cfg.features,
        "n_features": len(cfg.features),
        "backtest": report,
        "injury": inj,
        "importance": importance,
        "seconds": round(time.time() - t0, 1),
    }
    (MODEL_DIR / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    log(f"Done in {meta['seconds']}s")
    return meta


def load_meta() -> dict | None:
    f = MODEL_DIR / "meta.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else None


def load_models() -> dict:
    models: dict = {}
    for g in GROUPS:
        models[g] = {}
        for stat, _ in STAT_TARGETS[g]:
            models[g][stat] = lgb.Booster(model_file=str(MODEL_DIR / f"{g}_{stat}.txt"))
        for name, _, _ in POINT_TARGETS:
            models[g][name] = lgb.Booster(model_file=str(MODEL_DIR / f"{g}_{name}.txt"))
    return models


# ---------------------------------------------------------------- live predictions
def _next_game_availability(store: DataStore, fut: pd.DataFrame, table: dict) -> pd.Series:
    """Availability multiplier for each player's next game from the learned injury table.
    Uses this week's official report (status + practice) when published, else Sleeper's status."""
    out = pd.Series(np.nan, index=fut.index)
    if not table:
        return out
    lookup = {(r["status"], r["practice"]): r["multiplier"] for r in table["rows"]}
    first = fut.groupby("id")["week"].transform("min") == fut["week"]
    try:
        rep = sources.injury_reports(store.season, store.season)
        rep = rep[(rep["game_type"] == "REG") & (rep["week"] == store.current_week)].dropna(subset=["gsis_id"])
        official = {r.gsis_id: (r.report_status if isinstance(r.report_status, str) else "None",
                                PRACTICE.get(r.practice_status, "None")) for r in rep.itertuples()}
    except Exception:  # noqa: BLE001
        official = {}
    sleeper = store.players["injury_status"]
    for i in fut.index[first]:
        pid, wk = fut.at[i, "id"], fut.at[i, "week"]
        if wk == store.current_week and pid in official:
            st, pr = official[pid]
            if st in ("Questionable", "Doubtful", "Out", "None"):
                out[i] = lookup.get((st, pr), table["by_status"].get(st, 1.0))
        else:
            st = sleeper.get(pid)
            if st in ("Questionable", "Doubtful", "Out"):
                out[i] = table["by_status"].get(st)
    return out


def predict_future(store: DataStore, end_week: int = 18) -> pd.DataFrame:
    """Per-game predictions for every upcoming game: stats (scaled to the
    stats/direct ensemble), PPR median and calibrated floor/ceiling."""
    meta = load_meta() or {}
    cfg = Config.from_dict(meta["config"]) if meta.get("config") else Config()
    bt = meta.get("backtest", {})
    hist = D.load_history(store.season, log=lambda *_: None)
    ds = D.build(store, hist, end_week, log=lambda *_: None)
    fut = ds[ds["is_future"]]
    pred = predict_all(load_models(), fut, cfg)

    stats_ppr = stats_to_ppr(pred, fut["position"])
    a = float(bt.get("ensemble_alpha", 1.0))
    combined = a * stats_ppr + (1 - a) * pred["ppr"]
    scale = (combined / stats_ppr.clip(lower=0.5)).clip(0.7, 1.3)
    for _, targets in STAT_TARGETS.items():
        for stat, _ in targets:
            if stat in pred.columns and stat not in ("pass_att", "rush_att", "rec_tgt"):
                pred[stat] = pred[stat] * scale
    k = float(bt.get("range_scale", 1.0))
    med = combined
    pred["ppr_q10"] = (med - k * (pred["ppr"] - pred["ppr_q10"])).clip(lower=0)
    pred["ppr_q90"] = med + k * (pred["ppr_q90"] - pred["ppr"])
    pred["ppr"] = med
    pred["avail_ml"] = _next_game_availability(store, fut, meta.get("injury", {}))

    out = pd.concat([fut[["id", "week", "position"]], pred], axis=1).rename(columns={"ppr": "ppr_model"})
    return out.reset_index(drop=True)
