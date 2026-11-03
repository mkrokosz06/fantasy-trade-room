"""Player values (points over replacement) and lineup optimization.

Value = rest-of-season projected points minus what a replacement-level player
at the same position would score. Replacement level depends on league size and
starting slots, so a superflex or 3WR league automatically re-prices players.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .settings_store import SLOT_ELIGIBILITY

SKILL = ["QB", "RB", "WR", "TE"]
BENCH_DEPTH_SHARE = 0.5  # fraction of league bench spots that push replacement level deeper
SOFTPLUS_SCALE = 10.0    # smooths value near zero so depth players keep a little value


def ordered_slots(slots: dict) -> list[tuple[str, int]]:
    """Fillable slots, most restrictive first (QB before FLEX before SFLEX)."""
    fillable = [(s, int(n)) for s, n in slots.items() if s in SLOT_ELIGIBILITY and int(n) > 0]
    return sorted(fillable, key=lambda x: len(SLOT_ELIGIBILITY[x[0]]))


def replacement_levels(summary: pd.DataFrame, league: dict) -> dict[str, float]:
    teams = int(league.get("teams", 12))
    slots = league.get("slots", {})
    pool = summary.sort_values("ros_pts", ascending=False)
    taken: set = set()
    starters = {p: 0 for p in SKILL + ["K", "DST"]}
    for slot, n in ordered_slots(slots):
        elig = SLOT_ELIGIBILITY[slot]
        cand = pool[pool["position"].isin(elig) & ~pool.index.isin(taken)].head(n * teams)
        taken.update(cand.index)
        for p, c in cand["position"].value_counts().items():
            starters[p] = starters.get(p, 0) + int(c)

    bench_total = int(slots.get("BENCH", 0)) * teams * BENCH_DEPTH_SHARE
    skill_starters = sum(starters[p] for p in SKILL) or 1
    repl = {}
    for p in SKILL + ["K", "DST"]:
        extra = 0 if p in ("K", "DST") else math.ceil(bench_total * starters[p] / skill_starters)
        ranked = pool[pool["position"] == p]["ros_pts"].reset_index(drop=True)
        i = starters[p] + extra
        repl[p] = float(ranked.iloc[i]) if i < len(ranked) else 0.0
    return repl


def player_values(summary: pd.DataFrame, league: dict) -> tuple[pd.DataFrame, dict]:
    repl = replacement_levels(summary, league)
    out = summary.copy()
    out["replacement"] = out["position"].map(repl).fillna(0.0)
    out["vorp"] = (out["ros_pts"] - out["replacement"]).round(1)
    s = SOFTPLUS_SCALE
    out["value"] = (s * np.log1p(np.exp(out["vorp"].clip(upper=500) / s))).round(1)
    out["pos_rank"] = out.groupby("position")["ros_pts"].rank(ascending=False, method="first").astype(int)
    out["overall_rank"] = out["value"].rank(ascending=False, method="first").astype(int)
    return out, repl


def best_lineup_points(week_pts: dict[str, float], positions: dict[str, str], slots: dict) -> float:
    """Greedy optimal lineup for one week (optimal for nested slot eligibility)."""
    avail = sorted(week_pts.items(), key=lambda kv: -kv[1])
    used: set = set()
    total = 0.0
    for slot, n in ordered_slots(slots):
        elig = SLOT_ELIGIBILITY[slot]
        filled = 0
        for pid, pts in avail:
            if filled >= n:
                break
            if pid in used or positions.get(pid) not in elig:
                continue
            used.add(pid)
            total += pts
            filled += 1
    return total


class LineupEvaluator:
    """Rest-of-season optimal-lineup points for any set of player ids."""

    def __init__(self, weekly_proj: pd.DataFrame, slots: dict):
        self.slots = slots
        self.positions = dict(zip(weekly_proj["id"], weekly_proj["position"]))
        self.by_week: dict[int, dict[str, float]] = {}
        for wk, grp in weekly_proj.groupby("week"):
            self.by_week[int(wk)] = dict(zip(grp["id"], grp["pts"]))

    def total(self, roster: list[str] | set[str]) -> float:
        ids = set(roster)
        tot = 0.0
        for wk_pts in self.by_week.values():
            pts = {i: wk_pts[i] for i in ids if i in wk_pts}
            tot += best_lineup_points(pts, self.positions, self.slots)
        return round(tot, 1)
