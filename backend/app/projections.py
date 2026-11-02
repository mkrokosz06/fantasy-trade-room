"""Projection engine v1: recent usage x regressed efficiency x matchup.

For each player:
  1. Opportunity rates per game (pass attempts, carries, targets) from a
     recency-weighted average of their games (half-life in games, plus a
     discount for prior seasons).
  2. Efficiency per opportunity (yards/target, TD/carry, ...) shrunk toward the
     position average with empirical-Bayes style prior strength, so small
     samples and TD spikes regress.
  3. Per game: opponent factor (fantasy points allowed to the position vs league
     average, shrunk by games played) x Vegas factor (implied team total vs
     that team's average) applied to yardage/TD/reception stats.
  4. Blend with Sleeper's per-stat projections (more Sleeper weight when we
     have little data on a player, e.g. rookies), then apply injury availability.
  5. Score with the user's scoring settings.

This is deliberately transparent so the later trained model has a baseline
to beat.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import sources
from .data.store import DataStore
from .scoring import SLEEPER_MAP, STATS, score, to_canonical
from .settings_store import settings_key

OPP_STATS = ["pass_att", "rush_att", "rec_tgt"]
# (stat, opportunity denominator, prior strength in opportunities)
RATIOS = [
    ("pass_cmp", "pass_att", 100), ("pass_yd", "pass_att", 150),
    ("pass_td", "pass_att", 300), ("pass_int", "pass_att", 300),
    ("rush_yd", "rush_att", 80), ("rush_td", "rush_att", 150),
    ("rec", "rec_tgt", 50), ("rec_yd", "rec_tgt", 60), ("rec_td", "rec_tgt", 120),
]
# Stats projected directly per game, shrunk toward position mean (prior strength in games).
PER_GAME = [("pass_2pt", 6), ("rush_2pt", 6), ("rec_2pt", 6), ("st_td", 8), ("fum_lost", 8),
            ("fg_0_39", 4), ("fg_40_49", 4), ("fg_50p", 4), ("fg_miss", 6),
            ("xp_made", 4), ("xp_miss", 6)]
# Stats scaled by matchup/Vegas context (volume stays role-driven).
CONTEXT_STATS = ["pass_cmp", "pass_yd", "pass_td", "rush_yd", "rush_td", "rec", "rec_yd", "rec_td",
                 "fg_0_39", "fg_40_49", "fg_50p", "xp_made"]
SEASON_DISCOUNT = 0.7

# Availability multiplier by injury status for the player's next N games.
INJURY_AVAIL = {
    "Questionable": [0.85], "Doubtful": [0.25], "Out": [0.0, 0.85],
    "IR": [0.0, 0.0, 0.0, 0.0, 0.85], "PUP": [0.0, 0.0, 0.0, 0.0, 0.85],
    "NFI": [0.0, 0.0, 0.0, 0.0, 0.85], "Sus": [0.0, 0.0],
}


@dataclass
class Projections:
    weekly: pd.DataFrame      # id x future game: stats, factors, pts
    summary: pd.DataFrame     # one row per player: ros_pts, next_pts, ...
    base: pd.DataFrame        # per-game baseline (no matchup) per player
    priors: pd.DataFrame      # position efficiency priors (for expected points)
    matchups: pd.DataFrame    # defense x position factor table
    weeks: list[int]


def _weights(h: pd.DataFrame, season: int, half_life: float) -> pd.Series:
    order = h.groupby("id").cumcount(ascending=False)  # 0 = most recent game
    return (0.5 ** (order / half_life)) * (SEASON_DISCOUNT ** (season - h["season"]))


def build_baseline(store: DataStore, half_life: float):
    h = store.weekly.copy()
    pos_now = store.players["position"]
    h["position"] = h["id"].map(pos_now).fillna(h["position"])
    h["w"] = _weights(h, store.season, half_life)

    wsum = h[STATS].multiply(h["w"], axis=0)
    wsum["W"] = h["w"]
    wsum["id"] = h["id"]
    wsum["position"] = h["position"]
    g = wsum.groupby("id").agg({**{c: "sum" for c in STATS}, "W": "sum", "position": "last"})

    pos_tot = wsum.groupby("position")[STATS + ["W"]].sum()
    priors = pd.DataFrame(index=pos_tot.index)
    for stat, den, _ in RATIOS:
        priors[stat] = (pos_tot[stat] / pos_tot[den].replace(0, np.nan)).fillna(0)
    for stat, _ in PER_GAME:
        priors[stat + "_pg"] = pos_tot[stat] / pos_tot["W"]

    base = pd.DataFrame(index=g.index)
    base["position"] = g["position"]
    base["W"] = g["W"]
    for s in OPP_STATS:
        base[s] = g[s] / g["W"]
    pr = priors.reindex(g["position"]).set_axis(g.index)
    for stat, den, k in RATIOS:
        eff = (g[stat] + k * pr[stat]) / (g[den] + k)
        base[stat] = base[den] * eff
        base[stat + "_eff"] = eff
    for stat, k in PER_GAME:
        base[stat] = (g[stat] + k * pr[stat + "_pg"]) / (g["W"] + k)
    return base, priors, h


def matchup_table(h: pd.DataFrame, season: int, scoring: dict) -> pd.DataFrame:
    cur = h[h["season"] == season].copy()
    if cur.empty:
        return pd.DataFrame(columns=["opp", "position", "allowed", "games", "factor", "rank"])
    cur["pts"] = score(cur[STATS], scoring, cur["position"])
    per_game = cur.groupby(["opp", "position", "game_id"], as_index=False)["pts"].sum()
    d = per_game.groupby(["opp", "position"], as_index=False).agg(
        allowed=("pts", "mean"), games=("pts", "size"))
    lg = per_game.groupby("position")["pts"].mean()
    ratio = d["allowed"] / d["position"].map(lg)
    d["factor"] = (1 + (ratio - 1) * d["games"] / (d["games"] + 5)).clip(0.8, 1.2)
    # rank 1 = easiest matchup (most points allowed)
    d["rank"] = d.groupby("position")["allowed"].rank(ascending=False, method="min").astype(int)
    return d


def _availability(games: pd.DataFrame, status: pd.Series) -> pd.Series:
    idx = games.groupby("id").cumcount()
    st = games["id"].map(status)
    out = pd.Series(1.0, index=games.index)
    for s, mults in INJURY_AVAIL.items():
        for i, m in enumerate(mults):
            out[(st == s) & (idx == i)] = m
    return out


def _sleeper_frame(store: DataStore, weeks: list[int]) -> pd.DataFrame:
    frames = []
    for wk in weeks:
        try:
            df = sources.sleeper_projections(store.season, wk)
        except Exception:  # noqa: BLE001 - projections are optional
            continue
        if len(df):
            frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["id", "week"] + STATS)
    sp = pd.concat(frames, ignore_index=True)
    canon = to_canonical(sp, SLEEPER_MAP)
    sl_to_id = store.players.dropna(subset=["sleeper_id"]).set_index("sleeper_id")["id"]
    canon["id"] = sp["sleeper_id"].map(sl_to_id)
    canon["week"] = sp["week"].astype(int)
    return canon.dropna(subset=["id"]).drop_duplicates(["id", "week"])


def compute(store: DataStore, settings: dict) -> Projections:
    proj = settings["projection"]
    scoring = settings["scoring"]
    end_week = int(proj.get("end_week", 17))
    blend = float(proj.get("sleeper_blend", 0.35))

    base, priors, h = build_baseline(store, float(proj.get("half_life_games", 5)))
    mt = matchup_table(h, store.season, scoring)

    cw = store.current_week
    future = store.games[(~store.games["played"]) & store.games["week"].between(cw, end_week)]
    weeks = sorted(future["week"].unique().tolist())

    pl = store.players[store.players["team"].notna()][["id", "name", "position", "team", "injury_status"]]
    rows = pl.reset_index(drop=True).merge(
        future[["team", "week", "opp", "is_home", "implied", "gameday"]], on="team", how="inner")
    rows = rows.sort_values(["id", "week"]).reset_index(drop=True)

    # Context factors
    fac = mt.set_index(["opp", "position"])["factor"]
    rows["matchup"] = pd.Series(list(zip(rows["opp"], rows["position"]))).map(fac).fillna(1.0).values
    team_avg = store.games.groupby("team")["implied"].mean()
    vegas = (rows["implied"] / rows["team"].map(team_avg)) ** 0.5
    rows["vegas"] = vegas.clip(0.88, 1.12).fillna(1.0)
    rows["avail"] = _availability(rows, pl.set_index("id")["injury_status"]).values

    # Our model, per game
    ours = base.reindex(rows["id"])[STATS].set_axis(rows.index)
    ctx = rows["matchup"] * rows["vegas"]
    for s in CONTEXT_STATS:
        ours[s] = ours[s] * ctx
    have_ours = ours.notna().all(axis=1)

    # Sleeper, per game
    sl = _sleeper_frame(store, weeks)
    slm = rows[["id", "week"]].merge(sl, on=["id", "week"], how="left")[STATS].set_axis(rows.index)
    have_sl = slm.notna().any(axis=1)
    slm = slm.fillna(0.0)

    # Lean on Sleeper more when our sample is thin (< ~3 weighted games).
    conf = (base["W"].reindex(rows["id"]).fillna(0).values / 3.0).clip(0, 1)
    b = blend + (1 - blend) * (1 - conf)
    b = pd.Series(np.where(~have_ours, 1.0, np.where(~have_sl, 0.0, b)), index=rows.index)
    final = ours.fillna(0.0).multiply(1 - b, axis=0) + slm.multiply(b, axis=0)
    final = final.multiply(rows["avail"], axis=0)
    final[~(have_ours | have_sl)] = 0.0

    weekly = pd.concat([rows, final], axis=1)
    weekly["sleeper_weight"] = b.round(2)
    weekly["pts"] = score(weekly[STATS], scoring, weekly["position"])

    nxt = weekly.sort_values("week").groupby("id").head(1).set_index("id")
    summ = weekly.groupby("id").agg(
        name=("name", "first"), position=("position", "first"), team=("team", "first"),
        ros_pts=("pts", "sum"), games_left=("pts", "size"),
        healthy_games=("avail", lambda a: float((a > 0).sum())),
    )
    summ["proj_ppg"] = (summ["ros_pts"] / summ["healthy_games"].replace(0, np.nan)).fillna(0).round(2)
    summ["ros_pts"] = summ["ros_pts"].round(1)
    summ["next_week"] = nxt["week"]
    summ["next_opp"] = nxt["opp"]
    summ["next_pts"] = nxt["pts"]
    summ["next_matchup"] = nxt["matchup"].round(3)
    # Schedule strength: avg matchup factor of remaining games (>1 = easier)
    summ["sos"] = weekly.groupby("id")["matchup"].mean().round(3)

    return Projections(weekly=weekly, summary=summ, base=base, priors=priors,
                       matchups=mt, weeks=weeks)


_cache: dict[tuple, Projections] = {}
_lock = threading.Lock()


def get_projections(store: DataStore, settings: dict) -> Projections:
    key = (store.loaded_at, settings_key(settings))
    with _lock:
        if key not in _cache:
            _cache.clear()
            _cache[key] = compute(store, settings)
        return _cache[key]
