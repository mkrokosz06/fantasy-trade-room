"""User settings: scoring, league/roster format, projection knobs.

Saved to data/settings.json (gitignored). Defaults: 12-team redraft PPR.
"""
from __future__ import annotations

import copy
import json
import threading

from .config import SETTINGS_FILE
from .scoring import SCORING_PRESETS

# Which player positions each lineup slot accepts. Slots are filled in order of
# fewest eligible positions first (QB before FLEX before SUPERFLEX).
SLOT_ELIGIBILITY: dict[str, list[str]] = {
    "QB": ["QB"], "RB": ["RB"], "WR": ["WR"], "TE": ["TE"], "K": ["K"],
    "WRRB": ["RB", "WR"], "WRTE": ["WR", "TE"],
    "FLEX": ["RB", "WR", "TE"], "SFLEX": ["QB", "RB", "WR", "TE"],
}
SLOT_LABELS = {"QB": "QB", "RB": "RB", "WR": "WR", "TE": "TE", "K": "K", "DST": "D/ST",
               "WRRB": "RB/WR", "WRTE": "WR/TE", "FLEX": "FLEX (RB/WR/TE)",
               "SFLEX": "Superflex (QB/RB/WR/TE)", "BENCH": "Bench"}

_STANDARD_SLOTS = {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 1, "K": 1, "DST": 1, "BENCH": 7}

LEAGUE_PRESETS: dict[str, dict] = {
    "standard_12": {"label": "12-team, 1QB (ESPN default)", "teams": 12, "slots": dict(_STANDARD_SLOTS)},
    "standard_10": {"label": "10-team, 1QB", "teams": 10, "slots": dict(_STANDARD_SLOTS)},
    "three_wr_12": {"label": "12-team, 3WR", "teams": 12, "slots": {**_STANDARD_SLOTS, "WR": 3}},
    "superflex_12": {"label": "12-team Superflex", "teams": 12, "slots": {**_STANDARD_SLOTS, "SFLEX": 1}},
    "two_qb_12": {"label": "12-team 2QB", "teams": 12, "slots": {**_STANDARD_SLOTS, "QB": 2}},
}

DEFAULT_SETTINGS = {
    "scoring": {"preset": "ppr", **copy.deepcopy(SCORING_PRESETS["ppr"])},
    "league": {"preset": "standard_12", **copy.deepcopy(LEAGUE_PRESETS["standard_12"])},
    "projection": {
        # Weight given to Sleeper's projections when blending with our model (0-1).
        "sleeper_blend": 0.35,
        # Last week counted in rest-of-season (ROS) values. 17 = through fantasy playoffs.
        "end_week": 17,
        # Recency half-life in games for the usage model.
        "half_life_games": 5,
    },
    # ESPN league link. Login cookies are NOT stored here -- they live in .env.
    "espn": {"league_id": "", "my_team_id": ""},
}

_lock = threading.Lock()


def _merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict) and k not in ("rules", "slots"):
            out[k] = _merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_settings() -> dict:
    with _lock:
        if SETTINGS_FILE.exists():
            try:
                saved = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
                return _merge(DEFAULT_SETTINGS, saved)
            except (json.JSONDecodeError, OSError):
                pass
        return copy.deepcopy(DEFAULT_SETTINGS)


def save_settings(settings: dict) -> dict:
    merged = _merge(DEFAULT_SETTINGS, settings)
    with _lock:
        SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
        SETTINGS_FILE.write_text(json.dumps(merged, indent=2), encoding="utf-8")
    return merged


def settings_key(settings: dict) -> str:
    """Stable string used to cache computations per settings combination."""
    return json.dumps(settings, sort_keys=True)
