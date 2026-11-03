"""Track record for buy-low / sell-high / hold calls.

Each week the first time advice is computed, the calls are saved to
data/history/. Later, each saved call is graded against what the player
actually scored afterwards:

  BUY LOW   correct if they scored MORE than managers were seeing (perceived PPG)
  SELL HIGH correct if they scored LESS than that
  HOLD      correct if they stayed within 15% of it
"""
from __future__ import annotations

import json

import pandas as pd

from .config import DATA_DIR
from .scoring import STATS, score

HIST_DIR = DATA_DIR / "history"


def snapshot(signals: pd.DataFrame, season: int, after_week: int, engine: str) -> None:
    """after_week: the last week whose stats were known when the calls were made.
    Calls are graded only on games after it."""
    if signals is None or signals.empty:
        return
    HIST_DIR.mkdir(parents=True, exist_ok=True)
    f = HIST_DIR / f"calls_{season}_w{after_week:02d}_{engine}.json"
    if f.exists():
        return
    rows = signals.reset_index()[["id", "name", "position", "team", "call", "confidence",
                                  "perceived_ppg", "proj_ppg"]]
    f.write_text(json.dumps({"season": season, "after_week": after_week, "engine": engine,
                             "calls": json.loads(rows.to_json(orient="records"))}, indent=1), encoding="utf-8")


_bt_cache: dict = {}


def backtest_advisor(store, scoring: dict) -> dict:
    """Re-run the buy/sell/hold logic at past points (after weeks 5, 8 and 11 of
    the last two seasons) with only the data known then, and grade each call on
    the rest of that season. Uses the original engine's usage model as the
    projection, since that can be rebuilt point-in-time cheaply."""
    import numpy as np
    from types import SimpleNamespace

    from .advisor import EDGE_THRESHOLD, RELEVANT
    from .ml import dataset as D
    from .projections import build_baseline
    from .stats import expected_stats

    key = (store.loaded_at, json.dumps(scoring, sort_keys=True))
    if key in _bt_cache:
        return _bt_cache[key]
    hist = D.load_history(store.season, log=lambda *_: None)
    hist = hist.assign(pts=score(hist[STATS], scoring, hist["position"]))
    players = pd.DataFrame({"position": hist.groupby("id")["position"].last()})
    seasons = [store.season - 2, store.season - 1]
    graded, windows = [], []
    for S in seasons:
        for W in (5, 8, 11):
            known = hist[((hist["season"] < S) | ((hist["season"] == S) & (hist["week"] <= W)))
                         & (hist["season"] >= S - 2)]
            base, priors, _ = build_baseline(SimpleNamespace(weekly=known, players=players, season=S), 5.0)
            proj = score(base.reindex(columns=STATS).fillna(0), scoring, base["position"])
            cur = known[known["season"] == S].sort_values(["id", "week"]).copy()
            cur["xfp"] = score(expected_stats(cur, priors), scoring, cur["position"])
            cur["recent"] = cur.groupby("id").cumcount(ascending=False) < 3
            g = cur.groupby("id").agg(position=("position", "last"), name=("name", "last"),
                                      games=("week", "size"), ppg=("pts", "mean"))
            g["last3"] = cur[cur["recent"]].groupby("id")["pts"].mean()
            g["last3_xfp"] = cur[cur["recent"]].groupby("id")["xfp"].mean()
            g["prior_xfp"] = cur[~cur["recent"]].groupby("id")["xfp"].mean()
            g["proj"] = proj.reindex(g.index)
            g = g[(g["games"] >= 2) & g["position"].isin(RELEVANT) & g["proj"].notna()].copy()
            g["perceived"] = 0.5 * g["ppg"] + 0.5 * g["last3"].fillna(g["ppg"])
            lim = g["position"].map(RELEVANT)
            g = g[(g.groupby("position")["proj"].rank(ascending=False) <= lim)
                  | (g.groupby("position")["perceived"].rank(ascending=False) <= lim)].copy()
            g["edge"] = (g["proj"] - g["perceived"]) / g["perceived"].clip(lower=5)
            g["edge_adj"] = g["edge"] - g.groupby("position")["edge"].transform("median")
            trend = ((g["last3_xfp"] - g["prior_xfp"]) / g["prior_xfp"].clip(lower=3)).fillna(0).clip(-0.5, 0.5)
            g["score"] = g["edge_adj"] + 0.25 * trend
            g["call"] = np.where(g["score"] >= EDGE_THRESHOLD, "BUY LOW",
                                 np.where(g["score"] <= -EDGE_THRESHOLD, "SELL HIGH", "HOLD"))
            later = hist[(hist["season"] == S) & (hist["week"] > W)]
            g["actual"] = later.groupby("id")["pts"].mean().reindex(g.index)
            g = g.dropna(subset=["actual"])
            g["correct"] = np.where(g["call"] == "BUY LOW", g["actual"] > g["perceived"],
                                    np.where(g["call"] == "SELL HIGH", g["actual"] < g["perceived"],
                                             (g["actual"] - g["perceived"]).abs() <= 0.15 * g["perceived"].clip(lower=5)))
            g["season"], g["after_week"] = S, W
            graded.append(g)
            windows.append({"season": S, "after_week": W, "calls": int(len(g))})
    all_ = pd.concat(graded) if graded else pd.DataFrame()
    overall = {}
    for call, grp in all_.groupby("call") if len(all_) else []:
        overall[call] = {"n": int(len(grp)), "hit_rate": round(float(grp["correct"].mean()), 3),
                         "avg_change": round(float((grp["actual"] - grp["perceived"]).mean()), 2)}
    # Baseline for context: how often "scores more than managers are seeing"
    # happens for any graded player at all.
    base_rate = round(float((all_["actual"] > all_["perceived"]).mean()), 3) if len(all_) else None
    out = {"seasons": seasons, "windows": windows, "overall": overall, "base_rate_up": base_rate}
    _bt_cache.clear()
    _bt_cache[key] = out
    return out


def grade(store, scoring: dict, engine: str) -> dict:
    files = sorted(HIST_DIR.glob(f"calls_{store.season}_w*_{engine}.json")) if HIST_DIR.exists() else []
    w = store.weekly[store.weekly["season"] == store.season].copy()
    w["pts"] = score(w[STATS], scoring, w["position"])
    weeks = []
    for f in files:
        snap = json.loads(f.read_text(encoding="utf-8"))
        start = int(snap["after_week"])
        after = w[w["week"] > start]
        if after.empty:
            weeks.append({"week": start, "graded": False, "n": len(snap["calls"])})
            continue
        ppg = after.groupby("id")["pts"].mean()
        games = after.groupby("id").size()
        rows = []
        for c in snap["calls"]:
            if c["id"] not in ppg.index:
                continue
            actual, seen = float(ppg[c["id"]]), float(c["perceived_ppg"] or 0)
            if c["call"] == "BUY LOW":
                ok = actual > seen
            elif c["call"] == "SELL HIGH":
                ok = actual < seen
            else:
                ok = abs(actual - seen) <= 0.15 * max(seen, 5)
            rows.append({**c, "actual_ppg": round(actual, 2), "games_since": int(games[c["id"]]), "correct": bool(ok)})
        df = pd.DataFrame(rows)
        summary = {}
        for call, g in df.groupby("call") if len(df) else []:
            summary[call] = {"n": int(len(g)), "hit_rate": round(float(g["correct"].mean()), 3),
                             "avg_change": round(float((g["actual_ppg"] - g["perceived_ppg"]).mean()), 2)}
        weeks.append({"week": start, "graded": True, "through_week": int(after["week"].max()),
                      "summary": summary, "n": int(len(df))})
    return {"weeks": weeks, "tracking_since": weeks[0]["week"] if weeks else None}
