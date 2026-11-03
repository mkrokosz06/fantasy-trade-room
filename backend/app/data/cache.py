"""Download-and-cache helper. Every external fetch goes through here.

Files land in data/cache/. If a refresh fails (offline, API down) and a stale
copy exists, the stale copy is used so the app keeps working.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import httpx

from ..config import CACHE_DIR, CACHE_TTL_HOURS

_client = httpx.Client(
    follow_redirects=True,
    timeout=90,
    headers={"User-Agent": "fantasy-football-toolkit (personal project)"},
)


def is_fresh(path: Path, ttl_hours: float) -> bool:
    if not path.exists():
        return False
    if math.isinf(ttl_hours):
        return True
    return (time.time() - path.stat().st_mtime) < ttl_hours * 3600


def cached_download(url: str, name: str, ttl_hours: float | None = None,
                    force: bool = False, params: dict | None = None,
                    cookies: dict | None = None, headers: dict | None = None) -> Path:
    path = CACHE_DIR / name
    ttl = CACHE_TTL_HOURS if ttl_hours is None else ttl_hours
    if not force and is_fresh(path, ttl):
        return path
    try:
        resp = _client.get(url, params=params, cookies=cookies, headers=headers)
        resp.raise_for_status()
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_bytes(resp.content)
        tmp.replace(path)
    except httpx.HTTPError:
        if path.exists():
            return path  # stale but usable
        raise
    return path


def cached_json(url: str, name: str, ttl_hours: float | None = None,
                force: bool = False, params: dict | None = None,
                cookies: dict | None = None, headers: dict | None = None):
    path = cached_download(url, name, ttl_hours, force, params, cookies, headers)
    return json.loads(path.read_text(encoding="utf-8"))


def clear_cache(keep_immutable: bool = True) -> int:
    """Delete cached files. Prior-season nflverse files are kept by default."""
    n = 0
    for f in CACHE_DIR.iterdir():
        if keep_immutable and f.name.startswith("hist_"):
            continue
        f.unlink(missing_ok=True)
        n += 1
    return n
