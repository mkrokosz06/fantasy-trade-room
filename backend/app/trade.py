"""Trade evaluation.

Two lenses:
  * Value lens (always available): sums player values with a consolidation
    adjustment -- one great player beats two average ones because the extra
    player costs a roster spot and only one can start.
  * Lineup lens (needs rosters, e.g. from ESPN): projects each team's best
    starting lineup for every remaining week, before and after the trade.
"""
from __future__ import annotations

import pandas as pd

from .values import LineupEvaluator

CONSOLIDATION_P = 1.3  # p-norm exponent; 1.0 would be a plain sum


def side_value(values: list[float]) -> float:
    vals = [max(v, 0.0) for v in values]
    if not vals:
        return 0.0
    return round(sum(v ** CONSOLIDATION_P for v in vals) ** (1 / CONSOLIDATION_P), 1)


def verdict(a: float, b: float) -> dict:
    """Compare what side A receives (a) against what side B receives (b)."""
    top = max(a, b, 1e-9)
    diff_pct = (a - b) / top * 100
    mag = abs(diff_pct)
    if mag < 5:
        label, winner = "Fair trade", None
    elif mag < 15:
        label, winner = "Slight edge", "A" if diff_pct > 0 else "B"
    elif mag < 30:
        label, winner = "Clear edge", "A" if diff_pct > 0 else "B"
    else:
        label, winner = "Lopsided", "A" if diff_pct > 0 else "B"
    return {"label": label, "winner": winner, "diff_pct": round(diff_pct, 1)}


def evaluate(values: pd.DataFrame, gives_a: list[str], gives_b: list[str],
             evaluator: LineupEvaluator | None = None,
             roster_a: list[str] | None = None, roster_b: list[str] | None = None) -> dict:
    def players(ids):
        rows = []
        for pid in ids:
            if pid in values.index:
                r = values.loc[pid]
                rows.append({"id": pid, "name": r["name"], "position": r["position"],
                             "team": r["team"], "value": float(r["value"]),
                             "ros_pts": float(r["ros_pts"]), "proj_ppg": float(r["proj_ppg"])})
            else:
                rows.append({"id": pid, "name": pid, "position": None, "team": None,
                             "value": 0.0, "ros_pts": 0.0, "proj_ppg": 0.0})
        return rows

    pa, pb = players(gives_a), players(gives_b)
    # A receives what B gives, and vice versa.
    a_gets = side_value([p["value"] for p in pb])
    b_gets = side_value([p["value"] for p in pa])
    result = {
        "side_a": {"gives": pa, "receives_value": a_gets,
                   "raw_receives": round(sum(p["value"] for p in pb), 1)},
        "side_b": {"gives": pb, "receives_value": b_gets,
                   "raw_receives": round(sum(p["value"] for p in pa), 1)},
        "verdict": verdict(a_gets, b_gets),
    }
    # Suggest what would balance it: players near the gap in value.
    gap = abs(a_gets - b_gets)
    if gap > 3:
        used = set(gives_a) | set(gives_b)
        near = values[~values.index.isin(used) & values["position"].isin(["QB", "RB", "WR", "TE"])]
        near = near.iloc[(near["value"] - gap).abs().argsort()[:6]]
        result["balance"] = {
            "needed_value": round(gap, 1),
            "add_to": "A" if a_gets > b_gets else "B",  # side that should add a player
            "suggestions": [{"id": i, "name": r["name"], "position": r["position"],
                             "team": r["team"], "value": float(r["value"])}
                            for i, r in near.iterrows()],
        }
    if evaluator and roster_a is not None and roster_b is not None:
        result["lineup"] = lineup_impact(evaluator, roster_a, roster_b, gives_a, gives_b)
    return result


def lineup_impact(ev: LineupEvaluator, roster_a, roster_b, gives_a, gives_b) -> dict:
    ra, rb = set(roster_a), set(roster_b)
    na = (ra - set(gives_a)) | set(gives_b)
    nb = (rb - set(gives_b)) | set(gives_a)
    a0, a1, b0, b1 = ev.total(ra), ev.total(na), ev.total(rb), ev.total(nb)
    weeks = max(len(ev.by_week), 1)
    return {
        "weeks": weeks,
        "a": {"before": a0, "after": a1, "delta": round(a1 - a0, 1), "per_week": round((a1 - a0) / weeks, 2)},
        "b": {"before": b0, "after": b1, "delta": round(b1 - b0, 1), "per_week": round((b1 - b0) / weeks, 2)},
    }
