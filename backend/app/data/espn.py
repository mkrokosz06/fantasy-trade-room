"""ESPN fantasy league connector (unofficial read API).

Public leagues need only the league ID. Private leagues also need the
espn_s2 and SWID cookies from a logged-in browser session (see README).
"""
from __future__ import annotations

import httpx

from .. import config
from ..scoring import SCORING_PRESETS
from .cache import cached_json

BASE = "https://lm-api-reads.fantasy.espn.com/apis/v3/games/ffl/seasons/{season}/segments/0/leagues/{league_id}"

POSITION = {1: "QB", 2: "RB", 3: "WR", 4: "TE", 5: "K", 16: "DST"}
PRO_TEAMS = {
    0: "FA", 1: "ATL", 2: "BUF", 3: "CHI", 4: "CIN", 5: "CLE", 6: "DAL", 7: "DEN", 8: "DET",
    9: "GB", 10: "TEN", 11: "IND", 12: "KC", 13: "LV", 14: "LA", 15: "MIA", 16: "MIN",
    17: "NE", 18: "NO", 19: "NYG", 20: "NYJ", 21: "PHI", 22: "ARI", 23: "PIT", 24: "LAC",
    25: "SF", 26: "SEA", 27: "TB", 28: "WAS", 29: "CAR", 30: "JAX", 33: "BAL", 34: "HOU",
}
# ESPN lineup slot id -> our slot name
SLOTS = {0: "QB", 2: "RB", 3: "WRRB", 4: "WR", 5: "WRTE", 6: "TE", 7: "SFLEX",
         16: "DST", 17: "K", 20: "BENCH", 21: "IR", 23: "FLEX"}
# ESPN scoring stat id -> canonical stat. Unmapped ids are reported, not guessed.
STAT_IDS = {
    0: "pass_att", 1: "pass_cmp", 3: "pass_yd", 4: "pass_td", 19: "pass_2pt", 20: "pass_int",
    23: "rush_att", 24: "rush_yd", 25: "rush_td", 26: "rush_2pt",
    42: "rec_yd", 43: "rec_td", 44: "rec_2pt", 53: "rec", 58: "rec_tgt",
    72: "fum_lost",
    74: "fg_50p", 77: "fg_40_49", 80: "fg_0_39", 85: "fg_miss",
    86: "xp_made", 88: "xp_miss",
    198: "fg_50p",   # made FG 50-59 (newer ESPN leagues split 50+)
    101: "st_td",    # kickoff return TD
}
# Used only when the primary id above didn't set the stat. We project one
# 50+ FG bucket and one return-TD bucket, so 60+ FGs and punt return TDs
# borrow the 50-59 / kickoff values unless those are absent.
FALLBACK_IDS = {201: "fg_50p", 102: "st_td"}
# Team defense scoring ids.
DST_IDS = {"sack": 99, "int": 95, "fum_rec": 96, "safety": 98, "blk_kick": 97}
DST_TD_IDS = [93, 94, 103, 104, 101, 102]  # blocked-kick, defensive, INT/fumble return, kick/punt return TDs
# 18-21 and 22-27 points allowed score 0 in ESPN standard; their ids aren't
# confirmed, so they use placeholder ids that never match (always 0 points).
DST_PA_IDS = {89: (0, 0), 90: (1, 6), 91: (7, 13), 92: (14, 17), -1821: (18, 21), -2227: (22, 27),
              123: (28, 34), 124: (35, 45), 125: (46, 999)}
DST_YA_IDS = {128: (0, 99), 129: (100, 199), 130: (200, 299), 131: (300, 349), 132: (350, 399),
              133: (400, 449), 134: (450, 499), 135: (500, 549), 136: (550, 9999)}
# Distance-specific misses all collapse into our single fg_miss stat.
MISS_IDS = {76: "fg_miss", 79: "fg_miss", 82: "fg_miss"}


class ESPNError(Exception):
    pass


def dst_player_id(pro_team_id) -> str | None:
    team = PRO_TEAMS.get(pro_team_id)
    return f"DST:{team}" if team and team != "FA" else None


def fetch_league(league_id: str | None = None, season: int | None = None,
                 force: bool = False) -> dict:
    league_id = (league_id or config.ESPN_LEAGUE_ID).strip()
    if not league_id:
        raise ESPNError("No ESPN league ID set. Add ESPN_LEAGUE_ID to .env or enter it in the app.")
    cookies = {}
    if config.ESPN_S2 and config.ESPN_SWID:
        cookies = {"espn_s2": config.ESPN_S2, "SWID": config.ESPN_SWID}
    url = BASE.format(season=season, league_id=league_id)
    params = [("view", v) for v in ("mTeam", "mRoster", "mSettings", "mMatchupScore")]
    try:
        return cached_json(url, f"espn_league_{league_id}_{season}.json", ttl_hours=0.25,
                           force=force, params=params, cookies=cookies or None)
    except httpx.HTTPStatusError as e:
        code = e.response.status_code
        if code in (401, 403):
            raise ESPNError("ESPN says this league is private. Add ESPN_S2 and ESPN_SWID to .env "
                            "(see README) and restart the backend.") from e
        if code == 404:
            raise ESPNError(f"ESPN league {league_id} not found for {season}.") from e
        raise ESPNError(f"ESPN returned HTTP {code}.") from e
    except httpx.HTTPError as e:
        raise ESPNError(f"Could not reach ESPN: {e.__class__.__name__}") from e


def _cookies() -> dict | None:
    if config.ESPN_S2 and config.ESPN_SWID:
        return {"espn_s2": config.ESPN_S2, "SWID": config.ESPN_SWID}
    return None


def fetch_transactions(league_id: str, season: int, through_period: int) -> list[dict]:
    """Every trade-related transaction (proposals, accepts, declines, vetoes) this season."""
    url = BASE.format(season=season, league_id=league_id)
    out = []
    for p in range(1, through_period + 1):
        try:
            # Past periods don't change; the current one refreshes hourly.
            ttl = 1.0 if p >= through_period - 1 else 24 * 7
            data = cached_json(url, f"espn_tx_{league_id}_{season}_p{p}.json", ttl_hours=ttl,
                               params=[("view", "mTransactions2"), ("scoringPeriodId", p)], cookies=_cookies())
        except httpx.HTTPError:
            continue
        out += [t for t in data.get("transactions", []) if "TRADE" in (t.get("type") or "")]
    return out


def fetch_player_projections(league_id: str, season: int, scoring_period: int, force: bool = False) -> list[dict]:
    """ESPN's own projections for every relevant player, in this league's scoring.

    This is what leaguemates see in the ESPN app, so it's the best guess of how
    they value players when judging a trade.
    """
    import json as _json

    url = BASE.format(season=season, league_id=league_id)
    flt = {"players": {"filterStatus": {"value": ["FREEAGENT", "WAIVERS", "ONTEAM"]},
                       "filterSlotIds": {"value": [0, 2, 4, 6, 16, 17, 23]},
                       "limit": 600, "sortPercOwned": {"sortPriority": 1, "sortAsc": False}}}
    data = cached_json(url, f"espn_players_{league_id}_{season}_p{scoring_period}.json", ttl_hours=3, force=force,
                       params=[("view", "kona_player_info"), ("scoringPeriodId", scoring_period)],
                       cookies=_cookies(), headers={"X-Fantasy-Filter": _json.dumps(flt)})
    out = []
    for e in data.get("players", []):
        p = e.get("player") or {}
        rec = {"espn_id": str(p.get("id")), "name": p.get("fullName"),
               "dst_id": dst_player_id(p.get("proTeamId")) if p.get("defaultPositionId") == 16 else None,
               "position": POSITION.get(p.get("defaultPositionId")), "on_team_id": e.get("onTeamId") or 0,
               "status": e.get("status"), "pct_owned": (p.get("ownership") or {}).get("percentOwned"),
               "espn_ros": None, "espn_ros_avg": None, "espn_week": None, "espn_actual": None}
        for s in p.get("stats") or []:
            if s.get("seasonId") != season:
                continue
            src, split = s.get("statSourceId"), s.get("statSplitTypeId")
            if src == 1 and split == 0:     # rest-of-season projection
                rec["espn_ros"], rec["espn_ros_avg"] = s.get("appliedTotal"), s.get("appliedAverage")
            elif src == 1 and split == 1 and s.get("scoringPeriodId") == scoring_period:
                rec["espn_week"] = s.get("appliedTotal")
            elif src == 0 and split == 0:   # actual points so far
                rec["espn_actual"] = s.get("appliedTotal")
        r = (e.get("ratings") or {}).get("0") or {}
        rec["espn_pos_rank"] = r.get("positionalRanking")
        out.append(rec)
    return out


def parse_schedule(raw: dict) -> list[dict]:
    games = []
    for m in raw.get("schedule", []):
        home, away = m.get("home") or {}, m.get("away") or {}
        games.append({
            "period": m.get("matchupPeriodId"), "home": str(home.get("teamId")) if home else None,
            "away": str(away.get("teamId")) if away.get("teamId") is not None else None,
            "home_pts": home.get("totalPoints"), "away_pts": away.get("totalPoints"),
            "winner": m.get("winner"), "playoff": m.get("playoffTierType") not in (None, "NONE"),
        })
    return games


def parse_league(raw: dict, espn_to_id: dict[str, str]) -> dict:
    members = {m.get("id"): m for m in raw.get("members", [])}
    teams = []
    for t in raw.get("teams", []):
        name = t.get("name") or f"{t.get('location', '')} {t.get('nickname', '')}".strip()
        owners = [members.get(o, {}) for o in t.get("owners", [])]
        owner = ", ".join(filter(None, (f"{o.get('firstName', '')} {o.get('lastName', '')}".strip()
                                        or o.get("displayName") for o in owners)))
        roster = []
        for e in (t.get("roster") or {}).get("entries", []):
            p = (e.get("playerPoolEntry") or {}).get("player") or {}
            espn_id = str(p.get("id") or e.get("playerId"))
            pid = espn_to_id.get(espn_id)
            if POSITION.get(p.get("defaultPositionId")) == "DST":
                pid = dst_player_id(p.get("proTeamId"))
            roster.append({
                "espn_id": espn_id,
                "id": pid,
                "name": p.get("fullName"),
                "position": POSITION.get(p.get("defaultPositionId")),
                "pro_team": PRO_TEAMS.get(p.get("proTeamId")),
                "slot": SLOTS.get(e.get("lineupSlotId"), str(e.get("lineupSlotId"))),
            })
        rec = (t.get("record") or {}).get("overall") or {}
        teams.append({
            "team_id": str(t.get("id")), "name": name, "abbrev": t.get("abbrev"), "owner": owner,
            "wins": rec.get("wins"), "losses": rec.get("losses"), "ties": rec.get("ties"),
            "points_for": rec.get("pointsFor"), "roster": roster,
        })
    settings = raw.get("settings") or {}
    sched = settings.get("scheduleSettings") or {}
    return {
        "league_id": str(raw.get("id")), "season": raw.get("seasonId"),
        "name": settings.get("name"), "size": settings.get("size") or len(teams),
        "scoring_period": raw.get("scoringPeriodId"),
        "current_period": (raw.get("status") or {}).get("currentMatchupPeriod") or raw.get("scoringPeriodId"),
        "teams": teams,
        "settings": convert_settings(settings),
        "playoffs": {"regular_season_weeks": sched.get("matchupPeriodCount") or 14,
                     "teams": sched.get("playoffTeamCount") or 6,
                     "round_length": sched.get("playoffMatchupPeriodLength") or 1,
                     "seeding_tiebreak": sched.get("playoffSeedingRule")},
        "schedule": parse_schedule(raw),
    }


def convert_settings(settings: dict) -> dict:
    """ESPN league settings -> our league + scoring settings (with a report of what didn't map)."""
    counts = (settings.get("rosterSettings") or {}).get("lineupSlotCounts") or {}
    slots: dict[str, int] = {}
    unmapped_slots = []
    for sid, n in counts.items():
        n = int(n)
        if n <= 0:
            continue
        name = SLOTS.get(int(sid))
        if name is None:
            unmapped_slots.append(int(sid))
        elif name != "IR":
            slots[name] = slots.get(name, 0) + n

    items = (settings.get("scoringSettings") or {}).get("scoringItems") or []
    rules: dict[str, float] = {k: 0.0 for k in SCORING_PRESETS["ppr"]["rules"]}
    te_bonus = 0.0
    unmapped_stats = []
    miss_vals = []
    fallbacks: list[tuple[str, float]] = []
    for it in items:
        sid = int(it.get("statId"))
        pts = float(it.get("points", 0) or 0)
        if sid in MISS_IDS:
            miss_vals.append(pts)
            continue
        if sid in FALLBACK_IDS:
            fallbacks.append((FALLBACK_IDS[sid], pts))
            continue
        key = STAT_IDS.get(sid)
        if key is None:
            if pts:
                unmapped_stats.append({"stat_id": sid, "points": pts})
            continue
        rules[key] = pts
        overrides = it.get("pointsOverrides") or {}
        if key == "rec" and "6" in overrides:  # slot 6 = TE -> TE premium
            te_bonus = float(overrides["6"]) - pts
    for key, pts in fallbacks:
        if not rules.get(key):
            rules[key] = pts
    if miss_vals and not rules.get("fg_miss"):
        rules["fg_miss"] = sum(miss_vals) / len(miss_vals)

    # Team defense: start from all-zero tiers and fill in what the league scores.
    by_id = {int(it.get("statId")): float(it.get("points", 0) or 0) for it in items}
    dst = {k: by_id.get(sid, 0.0) for k, sid in DST_IDS.items()}
    dst["td"] = max([by_id.get(s, 0.0) for s in DST_TD_IDS] + [0.0])
    dst["pa_tiers"] = [[lo, hi, by_id.get(sid, 0.0)] for sid, (lo, hi) in DST_PA_IDS.items()]
    dst["ya_tiers"] = [[lo, hi, by_id.get(sid, 0.0)] for sid, (lo, hi) in DST_YA_IDS.items()]
    known_dst = set(DST_IDS.values()) | set(DST_TD_IDS) | set(DST_PA_IDS) | set(DST_YA_IDS)
    unmapped_stats = [u for u in unmapped_stats if u["stat_id"] not in known_dst]

    return {
        "league": {"preset": "espn", "label": f"ESPN: {settings.get('name', 'league')}",
                   "teams": int(settings.get("size") or 12), "slots": slots},
        "scoring": {"preset": "espn", "label": "ESPN league scoring", "rules": rules,
                    "te_rec_bonus": round(te_bonus, 2), "dst": dst},
        "unmapped": {"slots": unmapped_slots, "stats": unmapped_stats},
    }
