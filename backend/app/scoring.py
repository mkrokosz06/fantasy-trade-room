"""Canonical stat keys, source mappings, and fantasy scoring.

Every data source (nflverse actuals, Sleeper projections, our projection model)
is converted to the same canonical per-game stat columns, so one scoring
function works for all of them.
"""
from __future__ import annotations

import pandas as pd

# Canonical stat keys used everywhere in the app.
STATS = [
    "pass_att", "pass_cmp", "pass_yd", "pass_td", "pass_int", "pass_2pt",
    "rush_att", "rush_yd", "rush_td", "rush_2pt",
    "rec_tgt", "rec", "rec_yd", "rec_td", "rec_2pt",
    "fum_lost", "st_td",
    "fg_0_39", "fg_40_49", "fg_50p", "fg_miss", "xp_made", "xp_miss",
]

STAT_LABELS = {
    "pass_att": "Pass attempts", "pass_cmp": "Completions", "pass_yd": "Passing yards",
    "pass_td": "Passing TD", "pass_int": "Interceptions", "pass_2pt": "Passing 2PT",
    "rush_att": "Carries", "rush_yd": "Rushing yards", "rush_td": "Rushing TD",
    "rush_2pt": "Rushing 2PT", "rec_tgt": "Targets", "rec": "Receptions",
    "rec_yd": "Receiving yards", "rec_td": "Receiving TD", "rec_2pt": "Receiving 2PT",
    "fum_lost": "Fumbles lost", "st_td": "Return TD",
    "fg_0_39": "FG made 0-39", "fg_40_49": "FG made 40-49", "fg_50p": "FG made 50+",
    "fg_miss": "FG missed", "xp_made": "XP made", "xp_miss": "XP missed",
}

# nflverse stats_player_week column(s) -> canonical key (summed when a list).
NFLVERSE_MAP: dict[str, list[str]] = {
    "pass_att": ["attempts"], "pass_cmp": ["completions"], "pass_yd": ["passing_yards"],
    "pass_td": ["passing_tds"], "pass_int": ["passing_interceptions"],
    "pass_2pt": ["passing_2pt_conversions"],
    "rush_att": ["carries"], "rush_yd": ["rushing_yards"], "rush_td": ["rushing_tds"],
    "rush_2pt": ["rushing_2pt_conversions"],
    "rec_tgt": ["targets"], "rec": ["receptions"], "rec_yd": ["receiving_yards"],
    "rec_td": ["receiving_tds"], "rec_2pt": ["receiving_2pt_conversions"],
    "fum_lost": ["rushing_fumbles_lost", "receiving_fumbles_lost", "sack_fumbles_lost"],
    "st_td": ["special_teams_tds"],
    "fg_0_39": ["fg_made_0_19", "fg_made_20_29", "fg_made_30_39"],
    "fg_40_49": ["fg_made_40_49"],
    "fg_50p": ["fg_made_50_59", "fg_made_60_"],
    "fg_miss": ["fg_missed"], "xp_made": ["pat_made"], "xp_miss": ["pat_missed"],
}

# Sleeper projection stat key(s) -> canonical key.
SLEEPER_MAP: dict[str, list[str]] = {
    "pass_att": ["pass_att"], "pass_cmp": ["pass_cmp"], "pass_yd": ["pass_yd"],
    "pass_td": ["pass_td"], "pass_int": ["pass_int"], "pass_2pt": ["pass_2pt"],
    "rush_att": ["rush_att"], "rush_yd": ["rush_yd"], "rush_td": ["rush_td"],
    "rush_2pt": ["rush_2pt"], "rec_tgt": ["rec_tgt"], "rec": ["rec"],
    "rec_yd": ["rec_yd"], "rec_td": ["rec_td"], "rec_2pt": ["rec_2pt"],
    "fum_lost": ["fum_lost"], "st_td": ["pr_td", "def_kr_td"],
    "fg_0_39": ["fgm_0_19", "fgm_20_29", "fgm_30_39"], "fg_40_49": ["fgm_40_49"],
    "fg_50p": ["fgm_50p"], "fg_miss": ["fgmiss_0_19", "fgmiss_20_29", "fgmiss_30_39",
                                      "fgmiss_40_49", "fgmiss_50p"],
    "xp_made": ["xpm"], "xp_miss": ["xpmiss"],
}

_BASE_RULES = {
    "pass_yd": 0.04, "pass_td": 4, "pass_int": -2, "pass_2pt": 2,
    "rush_yd": 0.1, "rush_td": 6, "rush_2pt": 2,
    "rec": 1.0, "rec_yd": 0.1, "rec_td": 6, "rec_2pt": 2,
    "fum_lost": -2, "st_td": 6,
    "fg_0_39": 3, "fg_40_49": 4, "fg_50p": 5, "fg_miss": -1,
    "xp_made": 1, "xp_miss": -1,
}

SCORING_PRESETS: dict[str, dict] = {
    "ppr": {"label": "PPR", "rules": {**_BASE_RULES}, "te_rec_bonus": 0.0},
    "half_ppr": {"label": "Half-PPR", "rules": {**_BASE_RULES, "rec": 0.5}, "te_rec_bonus": 0.0},
    "standard": {"label": "Standard (non-PPR)", "rules": {**_BASE_RULES, "rec": 0.0}, "te_rec_bonus": 0.0},
    "ppr_6pt_pass": {"label": "PPR, 6-pt pass TD", "rules": {**_BASE_RULES, "pass_td": 6}, "te_rec_bonus": 0.0},
    "te_premium": {"label": "PPR + TE premium (1.5)", "rules": {**_BASE_RULES}, "te_rec_bonus": 0.5},
}


def to_canonical(df: pd.DataFrame, mapping: dict[str, list[str]]) -> pd.DataFrame:
    """Build canonical stat columns from a source frame (missing columns -> 0)."""
    out = {}
    for key, cols in mapping.items():
        present = [c for c in cols if c in df.columns]
        if present:
            out[key] = df[present].apply(pd.to_numeric, errors="coerce").fillna(0).sum(axis=1)
        else:
            out[key] = pd.Series(0.0, index=df.index)
    return pd.DataFrame(out, index=df.index)[STATS].astype(float)


def score(stats: pd.DataFrame, scoring: dict, positions: pd.Series | None = None) -> pd.Series:
    """Fantasy points for each row of a frame containing canonical stat columns."""
    rules = scoring.get("rules", {})
    pts = pd.Series(0.0, index=stats.index)
    for key, val in rules.items():
        if key in stats.columns and val:
            pts = pts + stats[key].fillna(0) * float(val)
    te_bonus = float(scoring.get("te_rec_bonus") or 0)
    if te_bonus and positions is not None and "rec" in stats.columns:
        pts = pts + (positions == "TE").astype(float) * stats["rec"].fillna(0) * te_bonus
    return pts.round(2)
