"""Paths and environment configuration."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

DATA_DIR = ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
SETTINGS_FILE = DATA_DIR / "settings.json"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ESPN league connection. espn_s2 / SWID are only needed for private leagues.
# They are login cookies -- keep them in .env, which is gitignored.
ESPN_LEAGUE_ID = os.getenv("ESPN_LEAGUE_ID", "").strip()
ESPN_S2 = os.getenv("ESPN_S2", "").strip()
ESPN_SWID = os.getenv("ESPN_SWID", "").strip()

# How long downloaded data is considered fresh before re-fetching.
CACHE_TTL_HOURS = float(os.getenv("CACHE_TTL_HOURS", "12"))

# Number of prior seasons loaded alongside the current one (for projections).
HISTORY_SEASONS = int(os.getenv("HISTORY_SEASONS", "2"))

FANTASY_POSITIONS = ("QB", "RB", "WR", "TE", "K")
