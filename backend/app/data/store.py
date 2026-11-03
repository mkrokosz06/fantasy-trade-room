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


TEAM_NICKS = {
    "ARI": "Cardinals", "ATL": "Falcons", "BAL": "Ravens", "BUF": "Bills", "CAR": "Panthers", "CHI": "Bears",
    "CIN": "Bengals", "CLE": "Browns", "DAL": "Cowboys", "DEN": "Broncos", "DET": "Lions", "GB": "Packers",
    "HOU": "Texans", "IND": "Colts", "JAX": "Jaguars", "KC": "Chiefs", "LA": "Rams", "LAC": "Chargers",
    "LV": "Raiders", "MIA": "Dolphins", "MIN": "Vikings", "NE": "Patriots", "NO": "Saints", "NYG": "Giants",
    "NYJ": "Jets", "PHI": "Eagles", "PIT": "Steelers", "SEA": "Seahawks", "SF": "49ers", "TB": "Buccaneers",
    "TEN": "Titans", "WAS": "Commanders",
}
ESPN_LOGO = {"LA": "lar", "WAS": "wsh"}


def dst_id(team: str) -> str:
    return f"DST:{team}"


@dataclass
class DataStore:
    season: int
    weekly: pd.DataFrame
    games: pd.DataFrame
    players: pd.DataFrame
    loaded_at: datetime
    warnings: list[str] = field(default_factory=list)
    dst_weekly: pd.DataFrame = field(default_factory=pd.DataFrame)  # actual team-defense results

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


def _dst_players() -> pd.DataFrame:
    rows = [{"id": dst_id(t), "name": f"{n} D/ST", "position": "DST", "team": t,
             "headshot": f"https://a.espncdn.com/i/teamlogos/nfl/500/{ESPN_LOGO.get(t, t.lower())}.png",
             "search_rank": 500} for t, n in TEAM_NICKS.items()]
    return pd.DataFrame(rows)


def _load_dst_weekly(season: int, warnings: list[str]) -> pd.DataFrame:
    """Actual team-defense results per game: sacks, takeaways, TDs, points and yards allowed."""
    sched = sources.schedules()
    sched = sched[sched["game_type"] == "REG"]
    frames = []
    for s in range(season - HISTORY_SEASONS, season + 1):
        try:
            ts = sources.team_stats_week(s, season)
        except Exception as e:  # noqa: BLE001
            warnings.append(f"No team stats for {s}: {e.__class__.__name__}")
            continue
        ts = ts[ts["season_type"] == "REG"].copy()
        num = lambda c: pd.to_numeric(ts[c], errors="coerce").fillna(0) if c in ts.columns else 0.0  # noqa: E731
        ts["yds_gained"] = num("passing_yards") - num("sack_yards_lost") + num("rushing_yards")
        offense = ts[["season", "week", "team", "yds_gained"]].rename(columns={"team": "opponent_team", "yds_gained": "dst_ya"})
        f = pd.DataFrame({
            "season": ts["season"].astype(int), "week": ts["week"].astype(int), "team": ts["team"],
            "opp": ts["opponent_team"], "game_id": ts["game_id"],
            "dst_sack": num("def_sacks"), "dst_int": num("def_interceptions"),
            "dst_fum_rec": num("fumble_recovery_opp"), "dst_safety": num("def_safeties"),
            "dst_blk": num("def_punt_blocks") + num("def_fg_blocks"),
            "dst_td": num("def_tds") + num("special_teams_tds"),
        })
        f = f.merge(offense, left_on=["season", "week", "opp"], right_on=["season", "week", "opponent_team"],
                    how="left").drop(columns=["opponent_team"])
        g = sched[sched["season"] == s]
        pa = pd.concat([
            pd.DataFrame({"game_id": g["game_id"], "team": g["home_team"], "dst_pa": g["away_score"]}),
            pd.DataFrame({"game_id": g["game_id"], "team": g["away_team"], "dst_pa": g["home_score"]}),
        ])
        f = f.merge(pa, on=["game_id", "team"], how="left")
        frames.append(f)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out["id"] = out["team"].map(dst_id)
    return out.dropna(subset=["dst_pa"])


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
            dst = _dst_players().set_index("id", drop=False)
            players = pd.concat([players, dst[~dst.index.isin(players.index)]])
            dst_weekly = _load_dst_weekly(season, warnings)
            _store = DataStore(season=season, weekly=weekly, games=games, players=players,
                               loaded_at=datetime.now(timezone.utc), warnings=warnings, dst_weekly=dst_weekly)
        return _store
