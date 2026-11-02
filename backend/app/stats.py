"""Stat tables built from nflverse weekly data, scored with the user's settings.

Includes "expected fantasy points" (xFP): what a player's opportunities
(pass attempts, carries, targets) are worth at league-average efficiency for
their position. Actual minus expected shows who is running hot or cold.
"""
from __future__ import annotations

import pandas as pd

from .data.store import DataStore
from .projections import PER_GAME, RATIOS, Projections
from .scoring import STATS, score

TD_STATS = ["pass_td", "rush_td", "rec_td"]


def expected_stats(rows: pd.DataFrame, priors: pd.DataFrame) -> pd.DataFrame:
    exp = pd.DataFrame(0.0, index=rows.index, columns=STATS)
    pr = priors.reindex(rows["position"]).set_axis(rows.index).fillna(0)
    for s in ("pass_att", "rush_att", "rec_tgt"):
        exp[s] = rows[s]
    for stat, den, _ in RATIOS:
        exp[stat] = rows[den] * pr[stat]
    for stat, _ in PER_GAME:
        exp[stat] = pr[stat + "_pg"]
    return exp


def scored_games(store: DataStore, proj: Projections, scoring: dict,
                 season: int | None = None) -> pd.DataFrame:
    w = store.weekly if season is None else store.weekly[store.weekly["season"] == season]
    w = scored_games_subset(store, proj, scoring, w)
    w["xtd"] = expected_stats(w, proj.priors)[TD_STATS].sum(axis=1)
    w["td"] = w[TD_STATS].sum(axis=1)
    w["opps"] = w["pass_att"] + w["rush_att"] + w["rec_tgt"]
    return w


def season_table(store: DataStore, proj: Projections, scoring: dict, season: int) -> pd.DataFrame:
    w = scored_games(store, proj, scoring, season).sort_values(["id", "week"])
    if w.empty:
        return pd.DataFrame()
    # Usage trends ignore games a player left early (injury exits), otherwise an
    # injury looks like a lost role. Points-based numbers keep them: that's what
    # trade partners see.
    typical = w.groupby("id")["opps"].transform("median")
    full = ~((typical >= 6) & (w["opps"] < 0.35 * typical))
    w["recent"] = w.groupby("id").cumcount(ascending=False) < 3
    w["recent_full"] = full & (w[full].groupby("id").cumcount(ascending=False) < 3).reindex(w.index, fill_value=False)
    g = w.groupby("id")
    t = g.agg(
        name=("name", "last"), position=("position", "last"), team=("team", "last"),
        games=("week", "size"), pts=("pts", "sum"), ppg=("pts", "mean"),
        xfp_ppg=("xfp", "mean"), td=("td", "sum"), xtd=("xtd", "sum"),
        pass_yd=("pass_yd", "sum"), pass_td=("pass_td", "sum"), pass_int=("pass_int", "sum"),
        rush_att=("rush_att", "sum"), rush_yd=("rush_yd", "sum"), rush_td=("rush_td", "sum"),
        rec_tgt=("rec_tgt", "sum"), rec=("rec", "sum"), rec_yd=("rec_yd", "sum"), rec_td=("rec_td", "sum"),
        target_share=("target_share", "mean"), wopr=("wopr", "mean"),
        air_yards_share=("air_yards_share", "mean"),
    )
    recent_pts = w[w["recent"]].groupby("id").agg(last3_ppg=("pts", "mean"))
    recent = w[w["recent_full"]].groupby("id").agg(
        last3_xfp=("xfp", "mean"), last3_tgt_share=("target_share", "mean"), last3_opps=("opps", "mean"))
    early = w[full & ~w["recent_full"]].groupby("id").agg(
        prior_xfp=("xfp", "mean"), prior_tgt_share=("target_share", "mean"), prior_opps=("opps", "mean"))
    t = t.join(recent_pts).join(recent).join(early)
    t["tgt_pg"] = t["rec_tgt"] / t["games"]
    t["car_pg"] = t["rush_att"] / t["games"]
    t["pos_rank"] = t.groupby("position")["pts"].rank(ascending=False, method="first").astype(int)
    for c in ["pts", "ppg", "xfp_ppg", "xtd", "last3_ppg", "last3_xfp", "prior_xfp",
              "tgt_pg", "car_pg", "last3_opps", "prior_opps"]:
        t[c] = t[c].round(2)
    for c in ["target_share", "wopr", "air_yards_share", "last3_tgt_share", "prior_tgt_share"]:
        t[c] = t[c].round(3)
    return t


def game_log(store: DataStore, proj: Projections, scoring: dict, pid: str) -> pd.DataFrame:
    w = store.weekly[store.weekly["id"] == pid]
    if w.empty:
        return w
    w = scored_games_subset(store, proj, scoring, w)
    cols = ["season", "week", "team", "opp", "pts", "xfp", "pass_att", "pass_cmp", "pass_yd",
            "pass_td", "pass_int", "rush_att", "rush_yd", "rush_td", "rec_tgt", "rec", "rec_yd",
            "rec_td", "fum_lost", "target_share", "air_yards_share", "wopr",
            "fg_0_39", "fg_40_49", "fg_50p", "fg_miss", "xp_made"]
    return w[cols].sort_values(["season", "week"], ascending=False)


def scored_games_subset(store, proj, scoring, w: pd.DataFrame) -> pd.DataFrame:
    w = w.copy()
    w["position"] = w["id"].map(store.players["position"]).fillna(w["position"])
    w["pts"] = score(w[STATS], scoring, w["position"])
    w["xfp"] = score(expected_stats(w, proj.priors), scoring, w["position"])
    return w


def defense_table(proj: Projections) -> pd.DataFrame:
    return proj.matchups.copy()
