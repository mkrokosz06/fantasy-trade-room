"""Loads and joins all source data into a few tidy in-memory tables.

    weekly    one row per player-game (regular season), canonical stats + usage
    games     one row per team-game for the current season (incl. future games)
    players   the player universe with cross-platform IDs and injury status
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone

import pandas as pd

from ..config import FANTASY_POSITIONS, HISTORY_SEASONS
from ..scoring import NFLVERSE_MAP, to_canonical
from . import sources

USAGE_COLS = ["target_share", "air_yards_share", "wopr", "receiving_air_yards",
              "passing_epa", "rushing_epa", "receiving_epa"]


@dataclass
class DataStore:
    season: int
    weekly: pd.DataFrame
    games: pd.DataFrame
    players: pd.DataFrame
    loaded_at: datetime
    warnings: list[str] = field(default_factory=list)

    @property
    def last_stats_week(self) -> int:
        cur = self.weekly[self.weekly["season"] == self.season]
        return int(cur["week"].max()) if len(cur) else 0

    @property
    def current_week(self) -> int:
        """First regular-season week that still has an unplayed game."""
        future = self.games[~self.games["played"]]
        return int(future["week"].min()) if len(future) else int(self.games["week"].max())


def _load_weekly(season: int, warnings: list[str], force: bool) -> pd.DataFrame:
    frames = []
    for s in range(season - HISTORY_SEASONS, season + 1):
        try:
            frames.append(sources.weekly_player_stats(s, season, force=force))
        except Exception as e:  # noqa: BLE001 - season may not exist yet
            warnings.append(f"No nflverse weekly stats for {s}: {e.__class__.__name__}")
    if not frames:
        raise RuntimeError("Could not load any nflverse weekly stats.")
    raw = pd.concat(frames, ignore_index=True)
    raw = raw[(raw["season_type"] == "REG") & raw["position"].isin(FANTASY_POSITIONS)].reset_index(drop=True)
    meta = raw[["player_id", "player_display_name", "position", "team", "opponent_team",
                "season", "week", "game_id", "headshot_url"]].rename(
        columns={"player_id": "id", "player_display_name": "name", "opponent_team": "opp"})
    usage = raw.reindex(columns=USAGE_COLS).apply(pd.to_numeric, errors="coerce").fillna(0.0)
    canon = to_canonical(raw, NFLVERSE_MAP)
    out = pd.concat([meta, usage, canon], axis=1)
    out["season"] = out["season"].astype(int)
    out["week"] = out["week"].astype(int)
    return out.sort_values(["season", "week"]).reset_index(drop=True)


def _load_games(season: int, force: bool) -> pd.DataFrame:
    g = sources.schedules(force=force)
    g = g[(g["season"] == season) & (g["game_type"] == "REG")].copy()
    total = pd.to_numeric(g["total_line"], errors="coerce")
    spread = pd.to_numeric(g["spread_line"], errors="coerce")  # + means home favored
    g["home_implied"] = total / 2 + spread / 2
    g["away_implied"] = total / 2 - spread / 2
    g["played"] = g["result"].notna()
    home = g.rename(columns={"home_team": "team", "away_team": "opp", "home_implied": "implied"})
    home["is_home"] = True
    away = g.rename(columns={"away_team": "team", "home_team": "opp", "away_implied": "implied"})
    away["is_home"] = False
    cols = ["game_id", "week", "gameday", "gametime", "team", "opp", "is_home", "implied", "played"]
    tg = pd.concat([home[cols], away[cols]], ignore_index=True)
    tg["week"] = tg["week"].astype(int)
    return tg.sort_values(["week", "team"]).reset_index(drop=True)


def _load_players(weekly: pd.DataFrame, season: int, force: bool) -> pd.DataFrame:
    sp = pd.DataFrame(list(sources.sleeper_players(force=force).values()))
    sp = sp[sp["position"].isin(FANTASY_POSITIONS)].copy()
    sp["sleeper_id"] = sp["player_id"].astype(str)
    sp["team"] = sp["team"].map(sources.fix_team)

    cw = sources.id_crosswalk(force=force)[["sleeper_id", "gsis_id", "espn_id"]]
    cw = cw.dropna(subset=["sleeper_id"]).drop_duplicates("sleeper_id")
    sp = sp.merge(cw.rename(columns={"gsis_id": "cw_gsis", "espn_id": "cw_espn"}),
                  on="sleeper_id", how="left")
    own_gsis = sp["gsis_id"].astype("string").str.strip().replace("", pd.NA)
    sp["gsis"] = sp["cw_gsis"].astype("string").fillna(own_gsis)
    own_espn = sp["espn_id"].astype("string").str.replace(r"\.0$", "", regex=True)
    sp["espn"] = own_espn.fillna(sp["cw_espn"].astype("string"))

    nfl = sources.nflverse_players(force=force)[["gsis_id", "espn_id", "headshot"]]
    nfl = nfl.dropna(subset=["gsis_id"]).drop_duplicates("gsis_id")
    sp = sp.merge(nfl.rename(columns={"gsis_id": "gsis", "espn_id": "nfl_espn"}), on="gsis", how="left")
    sp["espn"] = sp["espn"].fillna(sp["nfl_espn"].astype("string"))

    sp["id"] = sp["gsis"].where(sp["gsis"].notna(), "sl:" + sp["sleeper_id"])
    sp["name"] = sp["full_name"].fillna(sp["first_name"].fillna("") + " " + sp["last_name"].fillna(""))
    sp["headshot"] = sp["headshot"].fillna(
        "https://sleepercdn.com/content/nfl/players/thumb/" + sp["sleeper_id"] + ".jpg")
    players = sp[["id", "name", "position", "team", "sleeper_id", "espn", "age", "years_exp",
                  "injury_status", "injury_body_part", "headshot", "search_rank", "status"]]
    players = players.rename(columns={"espn": "espn_id"})

    # Anyone with stats this season but missing from Sleeper still belongs in the universe.
    cur = weekly[weekly["season"] == season].sort_values("week").groupby("id").tail(1)
    missing = cur[~cur["id"].isin(players["id"])]
    if len(missing):
        extra = missing[["id", "name", "position", "team", "headshot_url"]].rename(
            columns={"headshot_url": "headshot"})
        players = pd.concat([players, extra], ignore_index=True)

    has_stats = players["id"].isin(cur["id"])
    players = players[players["team"].notna() | has_stats].copy()
    players["search_rank"] = pd.to_numeric(players["search_rank"], errors="coerce").fillna(9_999_999)
    players = players.sort_values(["team", "search_rank"], na_position="last")
    players = players.drop_duplicates("id").set_index("id", drop=False)
    return players


_lock = threading.Lock()
_store: DataStore | None = None


def get_store(force: bool = False) -> DataStore:
    global _store
    with _lock:
        if _store is None or force:
            warnings: list[str] = []
            state = sources.sleeper_state(force=force)
            season = int(state.get("season") or state.get("league_season"))
            weekly = _load_weekly(season, warnings, force)
            games = _load_games(season, force)
            players = _load_players(weekly, season, force)
            _store = DataStore(season=season, weekly=weekly, games=games, players=players,
                               loaded_at=datetime.now(timezone.utc), warnings=warnings)
        return _store
