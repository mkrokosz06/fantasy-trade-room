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
}
# Distance-specific misses all collapse into our single fg_miss stat.
MISS_IDS = {76: "fg_miss", 79: "fg_miss", 82: "fg_miss"}


class ESPNError(Exception):
    pass


def fetch_league(league_id: str | None = None, season: int | None = None,
                 force: bool = False) -> dict:
    league_id = (league_id or config.ESPN_LEAGUE_ID).strip()
    if not league_id:
        raise ESPNError("No ESPN league ID set. Add ESPN_LEAGUE_ID to .env or enter it in the app.")
    cookies = {}
    if config.ESPN_S2 and config.ESPN_SWID:
        cookies = {"espn_s2": config.ESPN_S2, "SWID": config.ESPN_SWID}
    url = BASE.format(season=season, league_id=league_id)
    params = [("view", v) for v in ("mTeam", "mRoster", "mSettings")]
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
            roster.append({
                "espn_id": espn_id,
                "id": espn_to_id.get(espn_id),
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
    return {
        "league_id": str(raw.get("id")), "season": raw.get("seasonId"),
        "name": settings.get("name"), "size": settings.get("size") or len(teams),
        "scoring_period": raw.get("scoringPeriodId"),
        "teams": teams,
        "settings": convert_settings(settings),
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
    for it in items:
        sid = int(it.get("statId"))
        pts = float(it.get("points", 0) or 0)
        if sid in MISS_IDS:
            miss_vals.append(pts)
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
    if miss_vals and not rules.get("fg_miss"):
        rules["fg_miss"] = sum(miss_vals) / len(miss_vals)

    return {
        "league": {"preset": "espn", "label": f"ESPN: {settings.get('name', 'league')}",
                   "teams": int(settings.get("size") or 12), "slots": slots},
        "scoring": {"preset": "espn", "label": "ESPN league scoring", "rules": rules,
                    "te_rec_bonus": round(te_bonus, 2)},
        "unmapped": {"slots": unmapped_slots, "stats": unmapped_stats},
    }
