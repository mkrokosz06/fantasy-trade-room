"""Monte Carlo season simulator for an ESPN league.

Each simulated season:
  * every team's weekly score = its best projected lineup's points + random
    variation (per-player spread from the model's floor/ceiling range),
  * regular-season games follow the real remaining schedule (the current
    week starts from ESPN's live score),
  * standings by wins, then total points (ESPN's tiebreak),
  * the playoff bracket is played out week by week.

Trades are compared with the SAME random draws before and after, so the
difference in odds comes from the trade, not from simulation noise.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from ..settings_store import SLOT_ELIGIBILITY
from ..values import ordered_slots

N_SIMS = 5000
DST_MEAN, DST_SD = 6.5, 5.0      # team defenses aren't projected; same for every team
Z10 = 1.2816                      # 10th/90th percentile of a normal distribution


def _player_sd(pts: pd.Series, floor: pd.Series | None, ceiling: pd.Series | None) -> pd.Series:
    if floor is not None and ceiling is not None:
        sd = (ceiling - floor) / (2 * Z10)
        return sd.fillna(0.45 * pts + 1.5).clip(lower=0.5)
    # Original engine has no range: fantasy scoring spread is roughly proportional to the mean.
    return (0.45 * pts + 1.5).clip(lower=0.5)


def best_lineup(players: list[tuple[str, str, float]], slots: dict) -> list[str]:
    """Greedy best lineup: players = [(id, position, pts)] -> starting ids."""
    avail = sorted(players, key=lambda x: -x[2])
    used: set[str] = set()
    for slot, n in ordered_slots(slots):
        elig = SLOT_ELIGIBILITY[slot]
        filled = 0
        for pid, pos, _ in avail:
            if filled >= n:
                break
            if pid not in used and pos in elig:
                used.add(pid)
                filled += 1
    return list(used)


class LeagueSim:
    def __init__(self, weekly: pd.DataFrame, league: dict, slots: dict, seed: int = 11,
                 current_actuals: dict[str, float] | None = None):
        """current_actuals: points already scored this week by players whose games are done."""
        self.actuals = current_actuals or {}
        self.slots = slots
        self.teams = [t["team_id"] for t in league["teams"]]
        self.tix = {t: i for i, t in enumerate(self.teams)}
        po = league["playoffs"]
        self.reg_weeks = int(po["regular_season_weeks"])
        self.n_playoff = int(po["teams"])
        self.round_len = int(po.get("round_length") or 1)
        self.rounds = max(1, math.ceil(math.log2(max(self.n_playoff, 2))))
        last_week = self.reg_weeks + self.rounds * self.round_len

        sched = league["schedule"]
        self.current = int(league.get("current_period") or 1)
        open_reg = [g for g in sched if g["period"] <= self.reg_weeks and g["winner"] == "UNDECIDED" and g["away"]]
        first = min([g["period"] for g in open_reg], default=self.reg_weeks + 1)
        self.weeks = list(range(min(first, self.reg_weeks + 1), last_week + 1))
        self.wix = {w: i for i, w in enumerate(self.weeks)}
        self.games = [(self.wix[g["period"]], self.tix[g["home"]], self.tix[g["away"]])
                      for g in open_reg if g["home"] in self.tix and g["away"] in self.tix]

        # Current standings from completed games.
        T = len(self.teams)
        self.wins0 = np.zeros(T)
        self.pf0 = np.zeros(T)
        for t in league["teams"]:
            i = self.tix[t["team_id"]]
            self.wins0[i] = (t.get("wins") or 0) + 0.5 * (t.get("ties") or 0)
            self.pf0[i] = t.get("points_for") or 0
        self.in_progress = first == self.current

        w = weekly.copy()
        w["sd"] = _player_sd(w["pts"], w.get("floor"), w.get("ceiling"))
        self.wk = {wk: dict(zip(g["id"], zip(g["position"], g["pts"], g["sd"])))
                   for wk, g in w.groupby("week")}
        self.positions = dict(zip(w["id"], w["position"]))

        rng = np.random.default_rng(seed)
        self.z = rng.standard_normal((N_SIMS, T, len(self.weeks)))

        self.base_rosters: dict[str, list[str]] = {}
        self.current_starters: dict[str, list[str]] = {}
        for t in league["teams"]:
            self.base_rosters[t["team_id"]] = [p["id"] for p in t["roster"] if p["id"]]
            self.current_starters[t["team_id"]] = [p["id"] for p in t["roster"]
                                                   if p["id"] and p["slot"] not in ("BENCH", "IR")]
        self.base_params = {t: self.team_params(t, r) for t, r in self.base_rosters.items()}

    # ------------------------------------------------------------ team strength
    def team_params(self, team: str, roster: list[str]) -> tuple[np.ndarray, np.ndarray]:
        """Mean and SD of the team's score for every simulated week."""
        mean = np.zeros(len(self.weeks))
        var = np.zeros(len(self.weeks))
        last_known = max(self.wk) if self.wk else None
        for i, week in enumerate(self.weeks):
            wkp = self.wk.get(week) or (self.wk.get(last_known) if week > (last_known or 0) else {})
            if week == self.current and self.in_progress:
                # Week in progress: points already scored by the team's actual
                # starters + projections for starters who haven't played yet.
                starters = self.current_starters.get(team, [])
                done = sum(self.actuals.get(p, 0.0) for p in starters if p not in wkp)
                rem = [wkp[p] for p in starters if p in wkp]
                mean[i] = done + sum(x[1] for x in rem) + (DST_MEAN if self.slots.get("DST") else 0)
                var[i] = sum(x[2] ** 2 for x in rem) + (DST_SD ** 2 if self.slots.get("DST") else 0)
                continue
            cand = [(p, *wkp[p][:2]) for p in roster if p in wkp]
            starters = best_lineup(cand, self.slots)
            mean[i] = sum(wkp[p][1] for p in starters) + (DST_MEAN if self.slots.get("DST") else 0)
            var[i] = sum(wkp[p][2] ** 2 for p in starters) + (DST_SD ** 2 if self.slots.get("DST") else 0)
        return mean, np.sqrt(var)

    # ------------------------------------------------------------ simulate
    def run(self, overrides: dict[str, list[str]] | None = None) -> dict[str, dict]:
        params = dict(self.base_params)
        for t, roster in (overrides or {}).items():
            params[t] = self.team_params(t, roster)
        T = len(self.teams)
        mean = np.stack([params[t][0] for t in self.teams])          # T x W
        sd = np.stack([params[t][1] for t in self.teams])
        scores = mean[None] + sd[None] * self.z                       # N x T x W
        scores = np.maximum(scores, 0)

        wins = np.repeat(self.wins0[None], N_SIMS, axis=0)
        pf = np.repeat(self.pf0[None], N_SIMS, axis=0)
        for w, h, a in self.games:
            sh, sa = scores[:, h, w], scores[:, a, w]
            wins[:, h] += sh > sa
            wins[:, a] += sa > sh
            pf[:, h] += sh
            pf[:, a] += sa
        # Byes still add nothing; seeding by wins then points-for.
        key = wins * 100000 + pf
        order = np.argsort(-key, axis=1)                              # N x T, best first
        seed_of = np.empty_like(order)
        rows = np.arange(N_SIMS)[:, None]
        seed_of[rows, order] = np.arange(T)[None]                     # 0-based seed

        n = self.n_playoff
        size = 2 ** self.rounds
        bracket_seeds = _bracket_order(size)                          # 1-based seeds in slot order
        slots = np.full((N_SIMS, size), -1)
        for s_i, seed in enumerate(bracket_seeds):
            if seed <= n:
                slots[:, s_i] = order[:, seed - 1]
        for r in range(self.rounds):
            week = self.reg_weeks + 1 + r * self.round_len
            wi = self.wix.get(week, len(self.weeks) - 1)
            a, b = slots[:, 0::2], slots[:, 1::2]
            sa = np.where(a >= 0, scores[rows, np.maximum(a, 0), wi], -1)
            sb = np.where(b >= 0, scores[rows, np.maximum(b, 0), wi], -1)
            slots = np.where(sa >= sb, a, b)
        champ = slots[:, 0]

        out = {}
        byes = size - n  # top seeds that skip the first round
        for t, i in self.tix.items():
            seeds = seed_of[:, i]
            out[t] = {
                "playoff": float((seeds < n).mean()),
                "bye": float((seeds < byes).mean()) if byes else 0.0,
                "title": float((champ == i).mean()),
                "avg_wins": float(wins[:, i].mean()),
                "avg_seed": float(seeds.mean() + 1),
                "proj_week_pts": float(mean[i, [self.wix[w] for w in self.weeks if w <= self.reg_weeks]].mean())
                if any(w <= self.reg_weeks for w in self.weeks) else float(mean[i].mean()),
            }
        return out


def _bracket_order(size: int) -> list[int]:
    """Standard bracket slot order, e.g. 8 -> [1, 8, 4, 5, 2, 7, 3, 6]."""
    order = [1]
    while len(order) < size:
        m = len(order) * 2 + 1
        order = [x for s in order for x in (s, m - s)]
    return order


def situation(playoff: float) -> str:
    if playoff >= 0.75:
        return "Contender"
    if playoff >= 0.25:
        return "Bubble"
    return "Long shot"
