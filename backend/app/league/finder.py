"""Trade finder that ranks deals by what they do to playoff and title odds,
and by how likely the other manager is to accept.

1. Generate 1-for-1, 2-for-1 and 1-for-2 deals with every team.
2. Quick screen: keep deals that add to my projected starting lineup.
3. For the best ~120, simulate the season before/after for both teams and
   estimate acceptance.
4. Rank by (my playoff gain + 2x my title gain) weighted by how appealing
   the offer is to them -- a great trade they'll never accept is worth less
   than a good one they will.
"""
from __future__ import annotations

import itertools

from ..trade import side_value
from ..values import LineupEvaluator
from .context import LeagueContext

MAX_SIMULATED = 120


def find(L: LeagueContext, my_team: str, evaluator: LineupEvaluator, max_results: int = 25) -> list[dict]:
    vals = L.c.values["value"]

    def val(p):
        return float(vals.get(p, 0.0))

    mine_all = L.rosters[my_team]
    my_base = evaluator.total(mine_all)
    mine = sorted([p for p in mine_all if p in vals.index], key=val, reverse=True)[:12]
    cands = []
    for team, roster in L.rosters.items():
        if team == my_team:
            continue
        their_base = evaluator.total(roster)
        theirs = sorted([p for p in roster if p in vals.index], key=val, reverse=True)[:12]
        combos = [((a,), (b,)) for a in mine for b in theirs]
        combos += [((a1, a2), (b,)) for a1, a2 in itertools.combinations(mine[:8], 2) for b in theirs[:8]]
        combos += [((a,), (b1, b2)) for a in mine[:8] for b1, b2 in itertools.combinations(theirs[:8], 2)]
        for give, get in combos:
            vg, vr = side_value([val(p) for p in give]), side_value([val(p) for p in get])
            if vr > 0 and vg / vr < 0.5:
                continue  # wildly lopsided against them
            my_gain = evaluator.total((set(mine_all) - set(give)) | set(get)) - my_base
            if my_gain < 2:
                continue
            their_gain = evaluator.total((set(roster) - set(get)) | set(give)) - their_base
            if their_gain < -25:
                continue
            cands.append({"team": team, "give": list(give), "get": list(get),
                          "my_gain": my_gain, "their_gain": their_gain,
                          "quick": my_gain + 0.5 * min(their_gain, 15)})
    cands.sort(key=lambda x: -x["quick"])

    out = []
    for cnd in cands[:MAX_SIMULATED]:
        odds = L.trade_odds(my_team, cnd["team"], cnd["give"], cnd["get"])
        mine_d = odds["a"]["playoff_delta"] + 2 * odds["a"]["title_delta"]
        if mine_d <= 0.001:
            continue
        acc = L.acceptance(cnd["team"], they_get=cnd["give"], they_give=cnd["get"], odds=odds["b"])
        out.append({
            "team_id": cnd["team"], "team_name": L.teams[cnd["team"]]["name"],
            "give": cnd["give"], "get": cnd["get"],
            "my_gain": round(cnd["my_gain"], 1), "their_gain": round(cnd["their_gain"], 1),
            "odds": odds, "acceptance": acc,
            "score": round(mine_d * (0.35 + acc["appeal"]), 5),
        })
    out.sort(key=lambda x: -x["score"])
    # Variety: at most 2 deals for the same target player(s).
    seen: dict = {}
    result = []
    for r in out:
        k = tuple(r["get"])
        if seen.get(k, 0) >= 2:
            continue
        seen[k] = seen.get(k, 0) + 1
        result.append(r)
        if len(result) >= max_results:
            break
    return result
