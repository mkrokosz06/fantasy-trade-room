"""Raw fetchers for free data sources.

- nflverse (github.com/nflverse/nflverse-data): weekly player stats, schedules
  (incl. Vegas lines), player bios. Updated nightly in season.
- Sleeper (api.sleeper.app): current NFL week, injuries, per-stat projections.
- DynastyProcess (github.com/dynastyprocess/data): player ID crosswalk
  (nflverse gsis_id <-> sleeper_id <-> espn_id).
"""
from __future__ import annotations

import math

import pandas as pd

from .cache import cached_download, cached_json

NFLVERSE = "https://github.com/nflverse/nflverse-data/releases/download"
SLEEPER = "https://api.sleeper.app"
DP_IDS = "https://github.com/dynastyprocess/data/raw/master/files/db_playerids.csv"

# Sleeper (and ESPN) team codes -> nflverse codes.
TEAM_FIX = {"LAR": "LA", "OAK": "LV", "SD": "LAC", "STL": "LA", "WSH": "WAS", "JAC": "JAX"}


def fix_team(t):
    if t is None or (isinstance(t, float) and math.isnan(t)) or t == "":
        return None
    return TEAM_FIX.get(t, t)


# ---------------------------------------------------------------- nflverse
def weekly_player_stats(season: int, current_season: int, force: bool = False) -> pd.DataFrame:
    past = season < current_season
    name = f"{'hist_' if past else ''}stats_player_week_{season}.parquet"
    path = cached_download(f"{NFLVERSE}/stats_player/stats_player_week_{season}.parquet",
                           name, ttl_hours=math.inf if past else None, force=force and not past)
    return pd.read_parquet(path)


def schedules(force: bool = False) -> pd.DataFrame:
    path = cached_download(f"{NFLVERSE}/schedules/games.csv", "games.csv", ttl_hours=6, force=force)
    return pd.read_csv(path, low_memory=False)


def nflverse_players(force: bool = False) -> pd.DataFrame:
    path = cached_download(f"{NFLVERSE}/players/players.csv", "nflverse_players.csv",
                           ttl_hours=24, force=force)
    return pd.read_csv(path, low_memory=False, dtype={"espn_id": "string", "gsis_id": "string"})


def id_crosswalk(force: bool = False) -> pd.DataFrame:
    path = cached_download(DP_IDS, "db_playerids.csv", ttl_hours=24, force=force)
    df = pd.read_csv(path, low_memory=False, dtype=str)
    return df.replace({"NA": None, "": None})


def _season_parquet(release: str, fname: str, season: int, current_season: int,
                    base: str = NFLVERSE) -> pd.DataFrame:
    """Per-season parquet; past seasons are cached forever, the current one refreshes."""
    past = season < current_season
    name = f"{'hist_' if past else ''}{fname}"
    path = cached_download(f"{base}/{release}/{fname}", name,
                           ttl_hours=math.inf if past else None)
    return pd.read_parquet(path)


def snap_counts(season: int, current_season: int) -> pd.DataFrame:
    return _season_parquet("snap_counts", f"snap_counts_{season}.parquet", season, current_season)


def injury_reports(season: int, current_season: int) -> pd.DataFrame:
    return _season_parquet("injuries", f"injuries_{season}.parquet", season, current_season)


FFOPP = "https://github.com/ffverse/ffopportunity/releases/download"


def expected_points(season: int, current_season: int) -> pd.DataFrame:
    """ffopportunity: per player-game expected stats from a play-by-play model
    (accounts for field position, so it captures red-zone usage)."""
    return _season_parquet("latest-data", f"ep_weekly_{season}.parquet", season, current_season,
                           base=FFOPP)


def sleeper_projections_ppr(season: int, week: int) -> pd.DataFrame:
    """Historical Sleeper PPR projections, used only to benchmark the model."""
    rows = []
    for pos in ("QB", "RB", "WR", "TE"):
        data = cached_json(f"{SLEEPER}/projections/nfl/{season}/{week}",
                           f"hist_sleeper_proj_{season}_w{week}_{pos}.json", ttl_hours=math.inf,
                           params=[("season_type", "regular"), ("position[]", pos)])
        for r in data or []:
            st = r.get("stats") or {}
            if st.get("pts_ppr") is not None:
                rows.append({"sleeper_id": str(r.get("player_id")), "season": season, "week": week,
                             "sleeper_ppr": float(st["pts_ppr"])})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- Sleeper
def sleeper_state(force: bool = False) -> dict:
    return cached_json(f"{SLEEPER}/v1/state/nfl", "sleeper_state.json", ttl_hours=1, force=force)


def sleeper_players(force: bool = False) -> dict:
    # ~5 MB; Sleeper asks that this be fetched at most once a day.
    return cached_json(f"{SLEEPER}/v1/players/nfl", "sleeper_players.json", ttl_hours=24, force=force)


def sleeper_projections(season: int, week: int, force: bool = False) -> pd.DataFrame:
    params = [("season_type", "regular")] + [("position[]", p) for p in ("QB", "RB", "WR", "TE", "K")]
    rows = cached_json(f"{SLEEPER}/projections/nfl/{season}/{week}",
                       f"sleeper_proj_{season}_w{week}.json", ttl_hours=6, force=force,
                       params=params)
    recs = []
    for r in rows or []:
        stats = r.get("stats") or {}
        if not stats or not stats.get("gp", 1):
            continue
        recs.append({"sleeper_id": str(r.get("player_id")), "week": week, **stats})
    return pd.DataFrame(recs)
