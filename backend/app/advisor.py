"""Buy low / sell high / hold recommendations, and league trade finder.

Core idea: trade partners price players off what they *see* -- recent fantasy
points. We price them off what should happen next -- usage-driven projections.
The gap between the two is the trading edge.

  perceived_ppg = 50/50 blend of season PPG and last-3-games PPG (recency bias)
  projected_ppg = rest-of-season projection per healthy game
  edge          = (projected - perceived) / perceived

Supporting signals explain *why*: touchdown luck (TDs vs usage-expected TDs),
points vs expected points, usage trend (last 3 games vs earlier), remaining
schedule, injury status.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from .trade import side_value
from .values import LineupEvaluator

BUY, SELL, HOLD = "BUY LOW", "SELL HIGH", "HOLD"
EDGE_THRESHOLD = 0.12
# Trade finder: largest rest-of-season lineup loss we'd still propose to a partner
# (only when the deal also looks fair to them by recent points).
MAX_PARTNER_LOSS = 8.0
# Only grade fantasy-relevant players: top N at each position by projection or perception.
RELEVANT = {"QB": 24, "RB": 48, "WR": 60, "TE": 20}


def _ordinal(n: int) -> str:
    suf = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suf}"


def signals(season_tbl: pd.DataFrame, values: pd.DataFrame, players: pd.DataFrame) -> pd.DataFrame:
    if season_tbl.empty:
        return pd.DataFrame()
    df = values[values["position"].isin(RELEVANT)].join(
        season_tbl.drop(columns=["name", "position", "team", "pos_rank"]), how="inner")
    df = df[df["games"] >= 2].copy()
    df["perceived_ppg"] = (0.5 * df["ppg"] + 0.5 * df["last3_ppg"].fillna(df["ppg"])).round(2)
    df["edge"] = ((df["proj_ppg"] - df["perceived_ppg"]) / df["perceived_ppg"].clip(lower=5)).round(3)
    df["perf_gap"] = (df["ppg"] - df["xfp_ppg"]).round(2)          # + = running hot
    df["td_gap"] = (df["td"] - df["xtd"]).round(1)                  # + = TD luck
    df["usage_trend"] = ((df["last3_xfp"] - df["prior_xfp"]) / df["prior_xfp"].clip(lower=3)).round(3)
    df["injury_status"] = df.index.map(players["injury_status"])

    rank_proj = df.groupby("position")["proj_ppg"].rank(ascending=False)
    rank_perc = df.groupby("position")["perceived_ppg"].rank(ascending=False)
    lim = df["position"].map(RELEVANT)
    df = df[(rank_proj <= lim) | (rank_perc <= lim)].copy()

    df["sos_rank"] = df.groupby("position")["sos"].rank(ascending=False, method="min").astype(int)
    n_pos = df.groupby("position")["sos"].transform("size")

    # Projections regress everyone toward average, so raw edges skew negative.
    # Remove that position-wide bias, then let the recent usage trend (which the
    # projection only partly captures yet) nudge the score.
    df["edge_adj"] = df["edge"] - df.groupby("position")["edge"].transform("median")
    df["score"] = (df["edge_adj"] + 0.25 * df["usage_trend"].fillna(0).clip(-0.5, 0.5)).round(3)

    calls, reasons, conf = [], [], []
    for pid, r in df.iterrows():
        why = []
        edge = r["score"]
        if r["td_gap"] >= 2:
            why.append(f"{int(r['td'])} TDs vs {r['xtd']:.1f} expected from usage, likely to regress")
        elif r["td_gap"] <= -2:
            why.append(f"Only {int(r['td'])} TDs vs {r['xtd']:.1f} expected from usage, due for more")
        if r["perf_gap"] >= 3:
            why.append(f"Scoring {r['perf_gap']:.1f} PPG above what this usage normally produces")
        elif r["perf_gap"] <= -3:
            why.append(f"Scoring {abs(r['perf_gap']):.1f} PPG below what this usage normally produces")
        ut = r["usage_trend"]
        if pd.notna(ut) and ut >= 0.15:
            why.append(f"Usage rising: last 3 games worth {r['last3_xfp']:.1f} expected PPG vs {r['prior_xfp']:.1f} before")
        elif pd.notna(ut) and ut <= -0.15:
            why.append(f"Usage falling: last 3 games worth {r['last3_xfp']:.1f} expected PPG vs {r['prior_xfp']:.1f} before")
        if r["position"] in ("WR", "TE") and pd.notna(r["last3_tgt_share"]) and pd.notna(r["prior_tgt_share"]):
            d = r["last3_tgt_share"] - r["prior_tgt_share"]
            if abs(d) >= 0.05:
                why.append(f"Target share {r['prior_tgt_share']:.0%} -> {r['last3_tgt_share']:.0%} over last 3")
        n = int(n_pos[pid])
        if r["sos_rank"] <= max(3, n // 8):
            why.append(f"Remaining schedule is {_ordinal(int(r['sos_rank']))} easiest among graded {r['position']}s")
        elif r["sos_rank"] > n - max(3, n // 8):
            why.append(f"Remaining schedule is among the toughest for {r['position']}s")
        inj = r["injury_status"] if isinstance(r["injury_status"], str) else None
        if inj:
            why.append(f"Injury: {inj}. Projection already discounts missed games")

        # Confidence: saturating in edge size, discounted for small samples,
        # long injuries, and when the usage trend argues against the call.
        c = 1 - np.exp(-abs(edge) / 0.3)
        c *= min(1.0, (r["games"] / 6) ** 0.5)
        if inj in ("IR", "PUP", "NFI", "Sus"):
            c *= 0.75
        if edge >= EDGE_THRESHOLD:
            call = BUY
            why.insert(0, f"Projects {r['proj_ppg']:.1f} PPG rest of season vs {r['perceived_ppg']:.1f} PPG managers are seeing")
            if pd.notna(ut) and ut <= -0.15:
                c *= 0.6
                why.append("Risk: the role has shrunk lately. Make sure that isn't permanent before buying")
        elif edge <= -EDGE_THRESHOLD:
            call = SELL
            why.insert(0, f"Managers see {r['perceived_ppg']:.1f} PPG; projection is {r['proj_ppg']:.1f} PPG going forward")
            if pd.notna(ut) and ut >= 0.15:
                c *= 0.6
                why.append("Risk: the role is growing, which can justify the hot streak")
        else:
            call = HOLD
            why.insert(0, f"Projection ({r['proj_ppg']:.1f} PPG) roughly matches production ({r['perceived_ppg']:.1f} PPG); fairly priced")
            c = 1 - abs(edge) / EDGE_THRESHOLD
        calls.append(call)
        reasons.append(why)
        conf.append(float(np.clip(c, 0, 1)))

    df["call"] = calls
    df["reasons"] = reasons
    df["confidence"] = np.round(conf, 2)
    df["perceived_value"] = df["value"] * df["perceived_ppg"] / df["proj_ppg"].replace(0, np.nan)
    df["perceived_value"] = df["perceived_value"].fillna(df["value"]).round(1)
    return df.sort_values(["call", "score"], ascending=[True, False])


def find_trades(my_roster: list[str], other_rosters: dict[str, list[str]],
                values: pd.DataFrame, sig: pd.DataFrame, evaluator: LineupEvaluator,
                max_results: int = 25) -> list[dict]:
    """Search 1-for-1, 2-for-1 and 1-for-2 trades that improve my lineup.

    A trade is kept when it adds to my projected starting lineup AND has a
    realistic chance of being accepted: the partner's lineup barely gets
    worse (or improves), or the deal looks fair by the recent points they
    judge players on. Ranking rewards my gain, the partner's gain (win-win
    trades get accepted) and perceived fairness.
    """
    def val(pid):
        return float(values["value"].get(pid, 0.0))

    def pval(pid):
        if pid in sig.index:
            return float(sig.at[pid, "perceived_value"])
        return val(pid)

    my_base = evaluator.total(my_roster)
    mine = sorted([p for p in my_roster if p in values.index], key=val, reverse=True)[:12]
    results = []
    for team_id, roster in other_rosters.items():
        their_base = evaluator.total(roster)
        theirs = sorted([p for p in roster if p in values.index], key=val, reverse=True)[:12]
        combos = [((a,), (b,)) for a in mine for b in theirs]
        combos += [((a1, a2), (b,)) for a1, a2 in itertools.combinations(mine[:8], 2) for b in theirs[:8]]
        combos += [((a,), (b1, b2)) for a in mine[:8] for b1, b2 in itertools.combinations(theirs[:8], 2)]
        for give, get in combos:
            # Skip trades that are wildly lopsided by value -- nobody accepts them.
            vg, vr = side_value([val(p) for p in give]), side_value([val(p) for p in get])
            if vr > 0 and vg / vr < 0.6:
                continue
            new_mine = (set(my_roster) - set(give)) | set(get)
            my_gain = evaluator.total(new_mine) - my_base
            if my_gain < 3:
                continue
            new_theirs = (set(roster) - set(get)) | set(give)
            their_gain = evaluator.total(new_theirs) - their_base
            p_give = side_value([pval(p) for p in give])
            p_get = side_value([pval(p) for p in get])
            looks_fair = p_give >= 0.9 * p_get
            # Drop trades the partner would obviously reject.
            if their_gain < -MAX_PARTNER_LOSS or (their_gain < 0 and not looks_fair):
                continue
            score_ = my_gain + 0.75 * min(their_gain, 20) + (6 if looks_fair else 0)
            results.append({
                "team_id": team_id, "give": list(give), "get": list(get),
                "my_gain": round(my_gain, 1), "their_gain": round(their_gain, 1),
                "perceived_give": round(p_give, 1), "perceived_get": round(p_get, 1),
                "looks_fair_to_them": bool(looks_fair), "score": round(score_, 1),
            })
    results.sort(key=lambda r: -r["score"])
    # Avoid a list that is all the same player: cap repeats of any one target.
    out, seen = [], {}
    for r in results:
        k = tuple(r["get"])
        if seen.get(k, 0) >= 2:
            continue
        seen[k] = seen.get(k, 0) + 1
        out.append(r)
        if len(out) >= max_results:
            break
    return out
