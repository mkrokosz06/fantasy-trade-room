"""Offline unit tests for scoring, values, lineups, trades and ESPN settings mapping."""
import pandas as pd
import pytest

from app.data.espn import convert_settings
from app.scoring import NFLVERSE_MAP, SCORING_PRESETS, STATS, score, to_canonical
from app.settings_store import LEAGUE_PRESETS
from app.trade import side_value, verdict
from app.values import best_lineup_points, player_values, replacement_levels


def stat_row(**kw):
    return pd.DataFrame([{s: kw.get(s, 0.0) for s in STATS}])


def test_ppr_vs_half_vs_standard():
    row = stat_row(rec=10, rec_yd=100, rec_td=1)
    pos = pd.Series(["WR"])
    assert score(row, SCORING_PRESETS["ppr"], pos)[0] == pytest.approx(26)
    assert score(row, SCORING_PRESETS["half_ppr"], pos)[0] == pytest.approx(21)
    assert score(row, SCORING_PRESETS["standard"], pos)[0] == pytest.approx(16)


def test_te_premium_only_applies_to_te():
    row = pd.concat([stat_row(rec=10), stat_row(rec=10)], ignore_index=True)
    pts = score(row, SCORING_PRESETS["te_premium"], pd.Series(["TE", "WR"]))
    assert pts[0] == pytest.approx(15) and pts[1] == pytest.approx(10)


def test_qb_line():
    row = stat_row(pass_yd=300, pass_td=2, pass_int=1, rush_yd=20)
    assert score(row, SCORING_PRESETS["ppr"], pd.Series(["QB"]))[0] == pytest.approx(12 + 8 - 2 + 2)


def test_nflverse_mapping_sums_fumbles_and_fg_buckets():
    raw = pd.DataFrame([{"rushing_fumbles_lost": 1, "receiving_fumbles_lost": 1, "sack_fumbles_lost": 0,
                         "fg_made_0_19": 1, "fg_made_20_29": 1, "fg_made_30_39": 0,
                         "fg_made_50_59": 1, "fg_made_60_": 1}])
    c = to_canonical(raw, NFLVERSE_MAP)
    assert c.loc[0, "fum_lost"] == 2 and c.loc[0, "fg_0_39"] == 2 and c.loc[0, "fg_50p"] == 2
    assert c.loc[0, "pass_yd"] == 0  # missing columns -> 0


def _pool():
    rows = []
    for pos, n, top in [("QB", 30, 300), ("RB", 60, 250), ("WR", 70, 250), ("TE", 25, 180), ("K", 20, 120)]:
        for i in range(n):
            rows.append({"id": f"{pos}{i}", "name": f"{pos}{i}", "position": pos, "team": "X",
                         "ros_pts": top - i * 3.0, "proj_ppg": (top - i * 3.0) / 10})
    return pd.DataFrame(rows).set_index("id")


def test_superflex_raises_qb_value():
    pool = _pool()
    one_qb = replacement_levels(pool, LEAGUE_PRESETS["standard_12"])
    sf = replacement_levels(pool, LEAGUE_PRESETS["superflex_12"])
    assert sf["QB"] < one_qb["QB"]  # deeper replacement level -> QBs worth more
    v1, _ = player_values(pool, LEAGUE_PRESETS["standard_12"])
    v2, _ = player_values(pool, LEAGUE_PRESETS["superflex_12"])
    assert v2.loc["QB0", "value"] > v1.loc["QB0", "value"]


def test_lineup_greedy_fills_flex_with_best_remaining():
    pos = {"r1": "RB", "r2": "RB", "r3": "RB", "w1": "WR", "w2": "WR", "t1": "TE", "q1": "QB"}
    pts = {"r1": 20, "r2": 15, "r3": 14, "w1": 18, "w2": 10, "t1": 8, "q1": 22}
    slots = {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 1, "BENCH": 5}
    assert best_lineup_points(pts, pos, slots) == pytest.approx(22 + 20 + 15 + 18 + 10 + 8 + 14)


def test_consolidation_favors_the_stud():
    assert side_value([100]) > side_value([50, 50])
    assert side_value([50, 50]) > 50
    assert verdict(100, 98)["label"] == "Fair trade"
    assert verdict(150, 100)["winner"] == "A"


def test_espn_settings_conversion():
    settings = {
        "name": "Test League", "size": 10,
        "rosterSettings": {"lineupSlotCounts": {"0": 1, "2": 2, "4": 3, "6": 1, "23": 1, "17": 1,
                                                "16": 1, "20": 6, "21": 1, "7": 0}},
        "scoringSettings": {"scoringItems": [
            {"statId": 53, "points": 1.0, "pointsOverrides": {"6": 1.5}},
            {"statId": 4, "points": 6.0},
            {"statId": 3, "points": 0.04},
            {"statId": 999, "points": 2.0},
        ]},
    }
    out = convert_settings(settings)
    assert out["league"]["teams"] == 10
    assert out["league"]["slots"] == {"QB": 1, "RB": 2, "WR": 3, "TE": 1, "FLEX": 1, "K": 1,
                                      "DST": 1, "BENCH": 6}
    assert out["scoring"]["rules"]["rec"] == 1.0
    assert out["scoring"]["rules"]["pass_td"] == 6.0
    assert out["scoring"]["te_rec_bonus"] == pytest.approx(0.5)
    assert out["unmapped"]["stats"] == [{"stat_id": 999, "points": 2.0}]
