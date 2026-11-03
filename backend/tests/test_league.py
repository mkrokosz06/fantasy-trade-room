"""Season simulator and D/ST scoring tests (offline, synthetic league)."""
import pandas as pd
import pytest

from app.league.sim import LeagueSim, _bracket_order, situation
from app.scoring import score_dst


def test_bracket_order_gives_top_seeds_byes():
    assert _bracket_order(8) == [1, 8, 4, 5, 2, 7, 3, 6]
    assert _bracket_order(4) == [1, 4, 2, 3]


def _league(n_teams=6, playoff_teams=4, reg_weeks=4):
    teams = []
    for i in range(n_teams):
        teams.append({"team_id": str(i), "name": f"T{i}", "wins": 0, "losses": 0, "ties": 0, "points_for": 0,
                      "roster": [{"id": f"p{i}_{k}", "slot": "BENCH"} for k in range(3)]})
    sched = []
    for w in range(1, reg_weeks + 1):
        order = [str(i) for i in range(n_teams)]
        order = order[:1] + order[1:][w - 1:] + order[1:][:w - 1]   # simple rotation
        for a, b in zip(order[: n_teams // 2], reversed(order[n_teams // 2:])):
            sched.append({"period": w, "home": a, "away": b, "winner": "UNDECIDED", "home_pts": 0, "away_pts": 0})
    return {"teams": teams, "schedule": sched, "current_period": 1,
            "playoffs": {"regular_season_weeks": reg_weeks, "teams": playoff_teams, "round_length": 1}}


def _weekly(strength: dict[str, float], weeks):
    rows = []
    for t, s in strength.items():
        for k in range(3):
            for w in weeks:
                rows.append({"id": f"p{t}_{k}", "position": "RB", "week": w, "pts": s})
    return pd.DataFrame(rows)


def test_sim_probabilities_add_up_and_favor_stronger_team():
    lg = _league()
    strength = {str(i): 8 + 2 * i for i in range(6)}   # team 5 strongest
    sim = LeagueSim(_weekly(strength, range(1, 7)), lg, {"RB": 2, "BENCH": 1})
    odds = sim.run()
    assert sum(o["playoff"] for o in odds.values()) == pytest.approx(4.0, abs=1e-6)
    assert sum(o["title"] for o in odds.values()) == pytest.approx(1.0, abs=1e-6)
    assert odds["5"]["title"] > odds["0"]["title"]
    assert odds["5"]["playoff"] > odds["0"]["playoff"]


def test_trade_odds_use_same_random_draws():
    lg = _league()
    sim = LeagueSim(_weekly({str(i): 10 for i in range(6)}, range(1, 7)), lg, {"RB": 2, "BENCH": 1})
    # No-op "trade" must give exactly the same odds (common random numbers).
    assert sim.run({"0": sim.base_rosters["0"]}) == sim.run()


def test_situation_labels():
    assert situation(0.9) == "Contender" and situation(0.5) == "Bubble" and situation(0.1) == "Long shot"


def test_dst_scoring_standard_and_td_only():
    row = pd.DataFrame([{"dst_sack": 3, "dst_int": 1, "dst_fum_rec": 1, "dst_safety": 0, "dst_blk": 0,
                         "dst_td": 1, "dst_pa": 10, "dst_ya": 250}])
    # Standard: 3 + 2 + 2 + 6 + PA 7-13 (3) + YA 200-299 (2) = 18
    assert score_dst(row, None)[0] == pytest.approx(18)
    td_only = {"sack": 0, "int": 0, "fum_rec": 0, "safety": 0, "blk_kick": 0, "td": 6,
               "pa_tiers": [[0, 999, 0]], "ya_tiers": [[0, 9999, 0]]}
    assert score_dst(row, td_only)[0] == pytest.approx(6)
