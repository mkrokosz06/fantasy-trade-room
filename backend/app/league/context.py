"""League context: standings and odds, how leaguemates value players, how
active each manager is, trade history, and trade acceptance estimates.

"Perceived value" uses ESPN's own rest-of-season projections (in this league's
scoring): that's what leaguemates see in the ESPN app when judging an offer.
"""
from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
import pandas as pd

from ..data import espn
from ..scoring import STATS, score, score_dst
from ..trade import side_value
from ..values import player_values
from .sim import LeagueSim, situation


class LeagueContext:
    def __init__(self, c, league: dict, league_id: str):
        self.c = c
        self.league = league
        self.league_id = league_id
        self.teams = {t["team_id"]: t for t in league["teams"]}
        self.rosters = {t: [p["id"] for p in tm["roster"] if p["id"]] for t, tm in self.teams.items()}
        self.owner_of = {pid: t for t, r in self.rosters.items() for pid in r}

        # Points already scored this week (games played so far), in league scoring.
        cur = int(league.get("current_period") or c.store.current_week)
        w = c.store.weekly
        wk = w[(w["season"] == c.store.season) & (w["week"] == cur)]
        actuals = dict(zip(wk["id"], score(wk[STATS], c.settings["scoring"], wk["position"])))
        dw = c.store.dst_weekly
        if len(dw):
            dcur = dw[(dw["season"] == c.store.season) & (dw["week"] == cur)]
            actuals.update(zip(dcur["id"], score_dst(dcur, c.settings["scoring"].get("dst"))))
        self.sim = LeagueSim(c.proj.weekly, league, c.settings["league"]["slots"], current_actuals=actuals)
        self.odds = self.sim.run()

        self._load_espn_values()
        self._load_activity()

    # ------------------------------------------------------------ ESPN's view of player value
    def _load_espn_values(self):
        store = self.c.store
        period = int(self.league.get("scoring_period") or store.current_week)
        try:
            rows = espn.fetch_player_projections(self.league_id, store.season, period)
        except Exception:  # noqa: BLE001 - fall back to our own values
            rows = []
        espn_map = store.players.dropna(subset=["espn_id"]).set_index("espn_id")["id"].to_dict()
        df = pd.DataFrame(rows)
        self.espn = pd.DataFrame()
        if not df.empty:
            df["id"] = df["espn_id"].map(espn_map)
            if "dst_id" in df.columns:
                df["id"] = df["id"].fillna(df["dst_id"])
            df = df.dropna(subset=["id"]).drop_duplicates("id").set_index("id")
            self.espn = df
        # Perceived value: our value framework, but fed ESPN's projections.
        summ = self.c.values[["name", "position", "team", "ros_pts"]].copy()
        if not self.espn.empty:
            ros = self.espn["espn_ros"].reindex(summ.index)
            summ["ros_pts"] = ros.fillna(summ["ros_pts"])
        pv, _ = player_values(summ.assign(proj_ppg=0.0), self.c.settings["league"])
        self.perceived = pv["value"]

    def perceived_value(self, pid: str) -> float:
        return float(self.perceived.get(pid, self.c.values["value"].get(pid, 0.0)))

    # ------------------------------------------------------------ manager activity & history
    def _load_activity(self):
        period = int(self.league.get("scoring_period") or 1)
        try:
            tx = espn.fetch_transactions(self.league_id, self.c.store.season, period)
        except Exception:  # noqa: BLE001
            tx = []
        accepted = {t.get("relatedTransactionId") for t in tx if t.get("type") in ("TRADE_ACCEPT", "TRADE_UPHOLD")}
        vetoed = {t.get("relatedTransactionId") for t in tx if t.get("type") == "TRADE_VETO"}
        act = defaultdict(lambda: {"proposed": 0, "received": 0, "declined": 0, "completed": 0})
        self.history = []
        seen = set()
        espn_to_id = self.c.store.players.dropna(subset=["espn_id"]).set_index("espn_id")["id"].to_dict()
        for t in tx:
            if t.get("type") != "TRADE_PROPOSAL" or t.get("id") in seen:
                continue
            seen.add(t.get("id"))
            items = t.get("items") or []
            teams = {str(i.get("fromTeamId")) for i in items} | {str(i.get("toTeamId")) for i in items}
            proposer = str(t.get("teamId"))
            act[proposer]["proposed"] += 1
            for other in teams - {proposer}:
                act[other]["received"] += 1
            done = (t.get("status") == "EXECUTED" or t.get("id") in accepted) and t.get("id") not in vetoed
            if done:
                for tm in teams:
                    act[tm]["completed"] += 1
                sides = defaultdict(list)
                for i in items:
                    sides[str(i.get("toTeamId"))].append(espn_to_id.get(str(i.get("playerId")), str(i.get("playerId"))))
                self.history.append({"id": t.get("id"), "period": t.get("scoringPeriodId"), "proposer": proposer,
                                     "sides": dict(sides)})
        for t in tx:
            if t.get("type") == "TRADE_DECLINE":
                act[str(t.get("teamId"))]["declined"] += 1
        self.activity = {tid: dict(act[tid]) for tid in self.teams}

    def trade_history(self) -> list[dict]:
        """Completed trades and who has won them since, by actual fantasy points."""
        store, scoring = self.c.store, self.c.settings["scoring"]
        w = store.weekly[store.weekly["season"] == store.season].copy()
        w["pts"] = score(w[STATS], scoring, w["position"])
        out = []
        for h in self.history:
            after = w[w["week"] > (h["period"] or 0)]
            pts = after.groupby("id")["pts"].sum()
            sides = []
            for team, ids in h["sides"].items():
                sides.append({"team_id": team, "team_name": self.teams.get(team, {}).get("name", team),
                              "received": [{"id": i, "name": self._name(i), "pts_since": round(float(pts.get(i, 0.0)), 1),
                                            "ros_value": round(float(self.c.values["value"].get(i, 0.0)), 1)} for i in ids]})
            for s in sides:
                s["pts_since"] = round(sum(p["pts_since"] for p in s["received"]), 1)
                s["ros_value"] = round(sum(p["ros_value"] for p in s["received"]), 1)
            out.append({"period": h["period"], "sides": sides})
        return sorted(out, key=lambda x: -(x["period"] or 0))

    def _name(self, pid: str) -> str:
        p = self.c.store.players
        return str(p.at[pid, "name"]) if pid in p.index else pid

    # ------------------------------------------------------------ standings
    def standings(self) -> list[dict]:
        rows = []
        for tid, tm in self.teams.items():
            o = self.odds[tid]
            rows.append({"team_id": tid, "name": tm["name"], "owner": tm["owner"], "wins": tm.get("wins"),
                         "losses": tm.get("losses"), "ties": tm.get("ties"), "points_for": tm.get("points_for"),
                         **{k: round(v, 4) for k, v in o.items()}, "situation": situation(o["playoff"]),
                         "activity": self.activity.get(tid, {})})
        return sorted(rows, key=lambda r: (-r["playoff"], -r["title"]))

    # ------------------------------------------------------------ waiver wire
    def free_agents(self) -> list[str]:
        if self.espn.empty or "status" not in self.espn.columns:
            return []
        fa = self.espn[self.espn["status"].isin(["FREEAGENT", "WAIVERS"])]
        return [p for p in fa.index if p not in self.owner_of]

    def waiver_options(self, my_team: str, evaluator, top: int = 40, keep: int = 12) -> list[dict]:
        """Best free-agent pickups for my team: add the free agent, drop whichever
        of my lowest-value players hurts least, and simulate the season."""
        vals = self.c.values["value"]
        roster = self.rosters[my_team]
        base = evaluator.total(roster)
        # Players in an IR slot don't use a roster spot, so they're never the drop.
        on_ir = {p["id"] for p in self.teams[my_team]["roster"] if p.get("slot") == "IR"}
        droppable = sorted([p for p in roster if p not in on_ir], key=lambda p: float(vals.get(p, 0.0)))[:4]
        fas = sorted(self.free_agents(), key=lambda p: -float(vals.get(p, 0.0)))[:top]
        cands = []
        for fa in fas:
            best = None
            for d in droppable:
                gain = evaluator.total([p for p in roster if p != d] + [fa]) - base
                if best is None or gain > best[1]:
                    best = (d, gain)
            if best and best[1] > 0.5:
                cands.append((fa, best[0], best[1]))
        cands.sort(key=lambda x: -x[2])
        out = []
        for fa, drop, gain in cands[:keep]:
            after = self.sim.run({my_team: [p for p in roster if p != drop] + [fa]})[my_team]
            before = self.odds[my_team]
            e = self.espn.loc[fa] if fa in self.espn.index else None
            out.append({"add": fa, "drop": drop, "lineup_gain": round(gain, 1),
                        "playoff_before": before["playoff"], "playoff_after": after["playoff"],
                        "title_before": before["title"], "title_after": after["title"],
                        "status": None if e is None else e.get("status"),
                        "pct_owned": None if e is None else e.get("pct_owned")})
        out.sort(key=lambda r: -((r["playoff_after"] - r["playoff_before"]) + 2 * (r["title_after"] - r["title_before"])
                                 + r["lineup_gain"] / 1000))
        return out

    # ------------------------------------------------------------ trades
    def team_of(self, ids: list[str]) -> str | None:
        owners = {self.owner_of.get(i) for i in ids}
        return owners.pop() if len(owners) == 1 and None not in owners else None

    def trade_odds(self, team_a: str, team_b: str, gives_a: list[str], gives_b: list[str]) -> dict:
        ra = [p for p in self.rosters[team_a] if p not in gives_a] + list(gives_b)
        rb = [p for p in self.rosters[team_b] if p not in gives_b] + list(gives_a)
        after = self.sim.run({team_a: ra, team_b: rb})
        res = {}
        for side, t in (("a", team_a), ("b", team_b)):
            b0, b1 = self.odds[t], after[t]
            res[side] = {"team_id": t, "team_name": self.teams[t]["name"],
                         "situation": situation(b0["playoff"]),
                         **{f"{k}_before": round(b0[k], 4) for k in ("playoff", "bye", "title", "avg_wins")},
                         **{f"{k}_after": round(b1[k], 4) for k in ("playoff", "bye", "title", "avg_wins")}}
            res[side]["playoff_delta"] = round(b1["playoff"] - b0["playoff"], 4)
            res[side]["title_delta"] = round(b1["title"] - b0["title"], 4)
        return res

    def acceptance(self, receiver: str, they_get: list[str], they_give: list[str], odds: dict | None = None) -> dict:
        """Estimated chance the receiving team accepts, with the reasons.

        A heuristic, not a trained model -- there aren't enough trades in one
        league to learn from. Built from what drives real acceptances:
        how fair it looks in the ESPN app, whether it helps their team,
        their situation, and how often they trade.
        """
        reasons = []
        get_v = side_value([self.perceived_value(p) for p in they_get])
        give_v = side_value([self.perceived_value(p) for p in they_give])
        ratio = get_v / give_v if give_v > 0 else (2.0 if get_v > 0 else 1.0)
        score_ = 1 / (1 + math.exp(-5 * (ratio - 0.97)))
        if ratio >= 1.1:
            reasons.append(f"Looks like a win for them in the ESPN app ({get_v:.0f} vs {give_v:.0f} value)")
        elif ratio >= 0.92:
            reasons.append("Looks roughly even in the ESPN app, which is what they'll check")
        else:
            reasons.append(f"Looks like a loss for them in the ESPN app ({get_v:.0f} vs {give_v:.0f} value)")

        sit = situation(self.odds[receiver]["playoff"])
        if odds is not None:
            d = odds["playoff_delta"] + 2 * odds["title_delta"]
            score_ += float(np.clip(1.5 * d, -0.2, 0.2))
            if d > 0.02:
                reasons.append(f"Actually improves their team (playoff odds {odds['playoff_delta'] * 100:+.0f} pts)")
            elif d < -0.02:
                reasons.append(f"Actually hurts their team (playoff odds {odds['playoff_delta'] * 100:+.0f} pts)")
        if sit == "Long shot":
            score_ *= 0.7
            reasons.append("They're a long shot for the playoffs, so they may have checked out")
        elif sit == "Bubble":
            score_ *= 1.1
            reasons.append("They're on the playoff bubble: motivated to make moves")
        a = self.activity.get(receiver, {})
        moves = a.get("proposed", 0) + a.get("completed", 0)
        if moves == 0:
            score_ *= 0.8
            reasons.append("Hasn't been involved in any trades this season")
        elif moves >= 2:
            score_ *= 1.1
            reasons.append(f"Active trader ({a.get('proposed', 0)} offers sent, {a.get('completed', 0)} trades done)")
        appeal = float(np.clip(score_, 0.0, 1.0))
        # Scale by how often offers actually get accepted in this league (with a
        # prior of 1 in 4, typical for fantasy leagues).
        props = sum(x.get("proposed", 0) for x in self.activity.values())
        done = len(self.history)
        league_rate = (done + 2) / (props + 8)
        p = float(np.clip(appeal * league_rate / 0.25 * 0.6, 0.02, 0.9))
        label = "Strong offer" if appeal >= 0.65 else "Fair offer" if appeal >= 0.4 else "Long shot"
        return {"probability": round(p, 3), "appeal": round(appeal, 3), "label": label, "reasons": reasons,
                "perceived_get": round(get_v, 1), "perceived_give": round(give_v, 1),
                "league_accept_rate": {"proposals": props, "completed": done}}
