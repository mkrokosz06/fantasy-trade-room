"""Training / prediction dataset for the ML projection model.

One row per player-game (QB/RB/WR/TE, regular season, 2016 onward), plus
"future" rows for the current season's unplayed games. Every feature for a row
is computed only from games BEFORE it (shift-then-average), so historical rows
look exactly like what we'd have known at kickoff -- no leakage -- and future
rows get features from all games played so far.

Sources: nflverse weekly stats, snap counts, injury reports, schedules (Vegas
lines, home/away, roof, rest), player bios (age, draft pick); ffopportunity
expected points (play-by-play model incl. field position = red-zone usage).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..data import sources
from ..data.store import DataStore
from ..scoring import NFLVERSE_MAP, SCORING_PRESETS, STATS, score, to_canonical

FIRST_SEASON = 2016
POS = ["QB", "RB", "WR", "TE"]
POS_CODE = {"QB": 0, "RB": 1, "WR": 2, "TE": 3}
PPR = SCORING_PRESETS["ppr"]

EP_COLS = {"total_fantasy_points_exp": "xfp_ep", "pass_touchdown_exp": "xpass_td",
           "rush_touchdown_exp": "xrush_td", "rec_touchdown_exp": "xrec_td",
           "pass_yards_gained_exp": "xpass_yd", "rush_yards_gained_exp": "xrush_yd",
           "rec_yards_gained_exp": "xrec_yd"}
EP_TEAM = {"pass_attempt_team": "team_pass_att", "rush_attempt_team": "team_rush_att",
           "total_fantasy_points_exp_team": "team_xfp"}

# Player history columns that get rolling (pre-game) averages.
ROLL_SRC = ["pass_att", "rush_att", "rec_tgt", "target_share", "air_yards_share", "wopr",
            "snap_pct", "carry_share", "ppr", "pass_yd", "rush_yd", "rec_yd", "rec",
            "pass_td", "rush_td", "rec_td", "pass_int", "xfp_ep", "xpass_td", "xrush_td",
            "xrec_td", "xpass_yd", "xrush_yd", "xrec_yd"]
SPANS = (3, 10)

OUT_ALL_GAMES = {"IR", "PUP", "NFI", "Sus"}
OUT_NEXT_GAME = {"Out", "Doubtful"} | OUT_ALL_GAMES

FEATURES = (
    [f"{c}_e{s}" for s in SPANS for c in ROLL_SRC]
    + ["games_prior", "season_games_prior", "ppr_last", "prev_season_ppg",
       "team_pass_att_e", "team_rush_att_e", "team_xfp_e", "team_ppr_e",
       "opp_allowed", "is_home", "implied", "spread_team", "total_line", "dome", "rest",
       "age", "years_exp", "draft_pick", "own_status", "vacated_tgt", "vacated_car",
       "week", "pos_code"]
)

FEATURE_LABELS = {
    "ppr": "fantasy points", "xfp_ep": "expected fantasy points (play-by-play)",
    "rec_tgt": "targets", "rush_att": "carries", "pass_att": "pass attempts",
    "target_share": "target share", "carry_share": "carry share", "snap_pct": "snap share",
    "wopr": "WOPR", "air_yards_share": "air-yard share", "implied": "Vegas implied team total",
    "opp_allowed": "opponent points allowed to position", "vacated_tgt": "targets freed by injured teammates",
    "vacated_car": "carries freed by injured teammates", "team_xfp_e": "team offense strength",
    "prev_season_ppg": "last season PPG", "draft_pick": "draft pick", "years_exp": "years in league",
    "team_pass_att_e": "team pass attempts (recent)", "team_rush_att_e": "team carries (recent)",
    "team_ppr_e": "team fantasy points (recent)", "total_line": "Vegas game total",
    "spread_team": "point spread", "is_home": "home game", "own_status": "own injury status",
    "rest": "days of rest", "games_prior": "career games", "season_games_prior": "games this season",
    "ppr_last": "last game's points", "dome": "dome stadium", "week": "week of season",
    "pos_code": "WR or TE", "age": "age", "pass_yd": "passing yards", "rush_yd": "rushing yards",
    "rec_yd": "receiving yards", "rec": "receptions", "pass_td": "passing TDs", "rush_td": "rushing TDs",
    "rec_td": "receiving TDs", "pass_int": "interceptions", "xpass_td": "expected passing TDs",
    "xrush_td": "expected rushing TDs", "xrec_td": "expected receiving TDs",
    "xpass_yd": "expected passing yards", "xrush_yd": "expected rushing yards",
    "xrec_yd": "expected receiving yards",
}


def label_feature(f: str) -> str:
    for s in SPANS:
        suf = f"_e{s}"
        if f.endswith(suf):
            base = f[: -len(suf)]
            return f"{FEATURE_LABELS.get(base, base.replace('_', ' '))} (last ~{s} games)"
    return FEATURE_LABELS.get(f, f.replace("_", " "))


# ---------------------------------------------------------------- loading
def _ewm_prior(df: pd.DataFrame, key: str, cols: list[str], span: int) -> pd.DataFrame:
    """Exponentially weighted mean of each column over a group's PREVIOUS rows."""
    shifted = df.groupby(key, sort=False)[cols].shift(1)
    shifted[key] = df[key]
    out = (shifted.groupby(key, sort=False)[cols]
           .ewm(span=span, ignore_na=True).mean()
           .reset_index(level=0, drop=True))
    out = out.reindex(df.index)
    # ewm leaves leading NaN until a value is seen; forward-fill within group so
    # rows after a skipped (NaN) game keep the last known average.
    out[key] = df[key]
    return out.groupby(key, sort=False)[cols].ffill()


def load_history(current_season: int, log=print) -> pd.DataFrame:
    nfl_players = sources.nflverse_players()[["gsis_id", "pfr_id"]].dropna()
    pfr_to_gsis = dict(zip(nfl_players["pfr_id"], nfl_players["gsis_id"]))
    frames = []
    for s in range(FIRST_SEASON, current_season + 1):
        try:
            w = sources.weekly_player_stats(s, current_season)
        except Exception as e:  # noqa: BLE001
            log(f"  skipping {s}: no weekly stats ({e.__class__.__name__})")
            continue
        w = w[(w["season_type"] == "REG") & w["position"].isin(POS)].reset_index(drop=True)
        base = w[["player_id", "player_display_name", "position", "team", "opponent_team",
                  "season", "week", "game_id"]].rename(
            columns={"player_id": "id", "player_display_name": "name", "opponent_team": "opp"})
        usage = w.reindex(columns=["target_share", "air_yards_share", "wopr"]).astype(float)
        f = pd.concat([base, usage, to_canonical(w, NFLVERSE_MAP)], axis=1)

        try:
            ep = sources.expected_points(s, current_season)
            ep = ep[ep["player_id"].notna()][["player_id", "game_id", *EP_COLS, *EP_TEAM]]
            ep = ep.rename(columns={"player_id": "id", **EP_COLS, **EP_TEAM})
            f = f.merge(ep.drop_duplicates(["id", "game_id"]), on=["id", "game_id"], how="left")
        except Exception as e:  # noqa: BLE001
            log(f"  {s}: no expected-points data ({e.__class__.__name__})")
        try:
            sn = sources.snap_counts(s, current_season)
            sn = sn[sn["game_type"] == "REG"]
            sn = pd.DataFrame({"id": sn["pfr_player_id"].map(pfr_to_gsis), "game_id": sn["game_id"],
                               "snap_pct": sn["offense_pct"].astype(float)}).dropna(subset=["id"])
            f = f.merge(sn.drop_duplicates(["id", "game_id"]), on=["id", "game_id"], how="left")
        except Exception as e:  # noqa: BLE001
            log(f"  {s}: no snap counts ({e.__class__.__name__})")
        try:
            inj = sources.injury_reports(s, current_season)
            inj = inj[inj["game_type"] == "REG"].dropna(subset=["gsis_id", "week"])
            inj = pd.DataFrame({"id": inj["gsis_id"], "week": inj["week"].astype(int),
                                "report_status": inj["report_status"]})
            f = f.merge(inj.drop_duplicates(["id", "week"]), on=["id", "week"], how="left")
        except Exception as e:  # noqa: BLE001
            log(f"  {s}: no injury reports ({e.__class__.__name__})")
        frames.append(f)
        log(f"  loaded {s}: {len(f):,} player-games")
    hist = pd.concat(frames, ignore_index=True)
    for c in [*EP_COLS.values(), *EP_TEAM.values(), "snap_pct", "report_status"]:
        if c not in hist.columns:
            hist[c] = np.nan
    hist["season"] = hist["season"].astype(int)
    hist["week"] = hist["week"].astype(int)
    hist["ppr"] = score(hist[STATS], PPR, hist["position"])
    return hist


def _team_games(sched: pd.DataFrame) -> pd.DataFrame:
    g = sched[sched["game_type"] == "REG"].copy()
    g["total_line"] = pd.to_numeric(g["total_line"], errors="coerce")
    g["spread_line"] = pd.to_numeric(g["spread_line"], errors="coerce")
    g["dome"] = g["roof"].isin(["dome", "closed"]).astype(float)
    common = ["game_id", "season", "week", "gameday", "total_line", "dome"]
    home = g[common].assign(team=g["home_team"], opp=g["away_team"], is_home=1.0,
                            spread_team=g["spread_line"], rest=g["home_rest"])
    away = g[common].assign(team=g["away_team"], opp=g["home_team"], is_home=0.0,
                            spread_team=-g["spread_line"], rest=g["away_rest"])
    tg = pd.concat([home, away], ignore_index=True)
    tg["implied"] = tg["total_line"] / 2 + tg["spread_team"] / 2
    tg["season"] = tg["season"].astype(int)
    tg["week"] = tg["week"].astype(int)
    # Lines are only posted about a week ahead: fill future games with that
    # team's season averages so the model sees typical values, not blanks.
    for c in ("implied", "spread_team", "total_line"):
        tg[c] = tg[c].fillna(tg.groupby(["season", "team"])[c].transform("mean"))
        tg[c] = tg[c].fillna(tg.groupby("season")[c].transform("mean"))
    return tg


def _future_rows(store: DataStore, end_week: int) -> pd.DataFrame:
    games = store.games[(~store.games["played"]) & (store.games["week"] <= end_week)]
    pl = store.players[store.players["position"].isin(POS) & store.players["team"].notna()]
    pl = pl[~pl["id"].str.startswith("sl:")]
    rows = pl[["id", "name", "position", "team"]].reset_index(drop=True).merge(
        games[["team", "opp", "week", "game_id"]], on="team", how="inner")
    rows["season"] = store.season
    return rows


# ---------------------------------------------------------------- features
def build(store: DataStore, hist: pd.DataFrame, end_week: int = 18) -> pd.DataFrame:
    season_now = store.season
    fut = _future_rows(store, end_week)
    fut["is_future"] = True
    hist = hist.assign(is_future=False)
    df = pd.concat([hist, fut], ignore_index=True)
    df["t"] = df["season"] * 100 + df["week"]
    df = df.sort_values(["id", "t"]).reset_index(drop=True)
    df["pos_code"] = df["position"].map(POS_CODE).astype(float)

    # Games a player left early (or barely played) are noise for both
    # features and training labels.
    opps = df["pass_att"] + df["rush_att"] + df["rec_tgt"]
    typical = opps.where(~df["is_future"]).groupby(df["id"]).transform("median")
    df["partial"] = (~df["is_future"]) & (typical >= 6) & (opps < 0.35 * typical)

    df["carry_share"] = df["rush_att"] / df["team_rush_att"].replace(0, np.nan)
    src = df[ROLL_SRC].copy()
    src[df["partial"]] = np.nan
    src["id"] = df["id"]
    for s in SPANS:
        r = _ewm_prior(src, "id", ROLL_SRC, s)
        df[[f"{c}_e{s}" for c in ROLL_SRC]] = r[ROLL_SRC].to_numpy()

    played = (~df["is_future"] & ~df["partial"]).astype(float)
    df["games_prior"] = played.groupby(df["id"]).cumsum() - played
    df["season_games_prior"] = played.groupby([df["id"], df["season"]]).cumsum() - played
    last_ppr = df["ppr"].where(played > 0)
    df["ppr_last"] = last_ppr.groupby(df["id"]).shift(1)
    df["ppr_last"] = df.groupby("id")["ppr_last"].ffill()
    season_ppg = df[played > 0].groupby(["id", "season"])["ppr"].mean().rename("prev_season_ppg").reset_index()
    season_ppg["season"] += 1
    df = df.merge(season_ppg, on=["id", "season"], how="left")

    # ---- team volume / quality (pre-game rolling)
    tm = df[~df["is_future"]].groupby(["team", "season", "week"]).agg(
        team_pass_att=("team_pass_att", "first"), team_rush_att=("team_rush_att", "first"),
        team_xfp=("team_xfp", "first"), team_ppr=("ppr", "sum")).reset_index()
    tf = df[df["is_future"]][["team", "season", "week"]].drop_duplicates()
    tm = pd.concat([tm, tf], ignore_index=True).sort_values(["team", "season", "week"]).reset_index(drop=True)
    tcols = ["team_pass_att", "team_rush_att", "team_xfp", "team_ppr"]
    tr = _ewm_prior(tm, "team", tcols, 6)
    tm[[c + "_e" for c in tcols]] = tr[tcols].to_numpy()
    df = df.merge(tm[["team", "season", "week"] + [c + "_e" for c in tcols]],
                  on=["team", "season", "week"], how="left")

    # ---- opponent: PPR points allowed to the position, relative to league average
    dfn = df[~df["is_future"]]
    allowed = dfn.pivot_table(index=["opp", "season", "week"], columns="position", values="ppr",
                              aggfunc="sum").reset_index()
    of = df[df["is_future"]][["opp", "season", "week"]].drop_duplicates()
    allowed = pd.concat([allowed, of], ignore_index=True).sort_values(["opp", "season", "week"]).reset_index(drop=True)
    pcols = [p for p in POS if p in allowed.columns]
    league_avg = allowed[pcols].mean()
    ar = _ewm_prior(allowed, "opp", pcols, 10)
    for p in pcols:
        allowed[f"allow_{p}"] = ar[p].to_numpy() / league_avg[p]
    df = df.merge(allowed[["opp", "season", "week"] + [f"allow_{p}" for p in pcols]],
                  on=["opp", "season", "week"], how="left")
    df["opp_allowed"] = np.nan
    for p in pcols:
        m = df["position"] == p
        df.loc[m, "opp_allowed"] = df.loc[m, f"allow_{p}"]
    df = df.drop(columns=[f"allow_{p}" for p in pcols])

    # ---- game context from schedules
    tg = _team_games(sources.schedules())
    df = df.merge(tg[["team", "season", "week", "gameday", "is_home", "implied", "spread_team",
                      "total_line", "dome", "rest"]], on=["team", "season", "week"], how="left")

    # ---- player bio
    bio = sources.nflverse_players()[["gsis_id", "birth_date", "draft_pick", "rookie_season"]]
    bio = bio.dropna(subset=["gsis_id"]).drop_duplicates("gsis_id").rename(columns={"gsis_id": "id"})
    df = df.merge(bio, on="id", how="left")
    gd = pd.to_datetime(df["gameday"], errors="coerce")
    df["age"] = (gd - pd.to_datetime(df["birth_date"], errors="coerce")).dt.days / 365.25
    df["years_exp"] = df["season"] - pd.to_numeric(df["rookie_season"], errors="coerce")
    df["draft_pick"] = pd.to_numeric(df["draft_pick"], errors="coerce").fillna(300)

    # ---- own injury status this week (history: report; future: Sleeper, next game only)
    status_code = {"Questionable": 1.0, "Doubtful": 2.0}
    df["own_status"] = df["report_status"].map(status_code).fillna(0.0)
    if df["is_future"].any():
        sl = store.players["injury_status"]
        first_future = df["is_future"] & (df.groupby("id")["is_future"].cumsum() == 1)
        df.loc[first_future, "own_status"] = df.loc[first_future, "id"].map(sl).map(status_code).fillna(0.0)

    df = _vacated(df, store)
    df = df.sort_values(["id", "t"]).reset_index(drop=True)
    return df


def _vacated(df: pd.DataFrame, store: DataStore) -> pd.DataFrame:
    """Targets/carries freed up by teammates who aren't playing.

    Candidates for a team's game = players who appeared for that team in any of
    its previous 3 games. Their recent target/carry share is "vacated" if they
    don't play (history) or are listed out / changed teams (future games).
    """
    share = df[["id", "target_share", "carry_share"]].copy()
    share[df["partial"]] = np.nan
    share["ts"] = share.groupby("id")["target_share"].transform(lambda s: s.ewm(span=6, ignore_na=True).mean())
    share["cs"] = share.groupby("id")["carry_share"].transform(lambda s: s.ewm(span=6, ignore_na=True).mean())
    df = df.assign(_ts=share["ts"].fillna(0), _cs=share["cs"].fillna(0))

    tgames = df[["team", "t", "is_future"]].drop_duplicates(["team", "t"]).sort_values(["team", "t"])
    tgames["k"] = tgames.groupby("team").cumcount()
    df = df.merge(tgames[["team", "t", "k"]], on=["team", "t"], how="left")

    hist = df[~df["is_future"]][["id", "team", "k", "_ts", "_cs"]]
    exp = pd.concat([hist.assign(m=hist["k"] + o) for o in (1, 2, 3)], ignore_index=True)
    exp = exp.sort_values("k").drop_duplicates(["id", "team", "m"], keep="last")
    played = set(zip(hist["id"], hist["team"], hist["k"]))
    exp = exp[[(i, t, m) not in played for i, t, m in zip(exp["id"], exp["team"], exp["m"])]]
    # restrict to team-games that are historical (future handled below)
    hist_k = tgames[~tgames["is_future"]][["team", "k"]].rename(columns={"k": "m"})
    exp_h = exp.merge(hist_k, on=["team", "m"], how="inner")
    vac_h = exp_h.groupby(["team", "m"])[["_ts", "_cs"]].sum()

    # Future: candidates from each team's last 3 played games, absent if listed out or moved teams.
    vac_f = []
    fut_k = tgames[tgames["is_future"]]
    if len(fut_k):
        status = store.players["injury_status"]
        cur_team = store.players["team"]
        for team, fk in fut_k.groupby("team"):
            last_k = tgames[(tgames["team"] == team) & ~tgames["is_future"]]["k"].max()
            if pd.isna(last_k):
                continue
            cand = hist[(hist["team"] == team) & (hist["k"] > last_k - 3)]
            cand = cand.sort_values("k").drop_duplicates("id", keep="last")
            st = cand["id"].map(status)
            moved = cand["id"].map(cur_team) != team
            for i, m in enumerate(sorted(fk["k"])):
                out_set = OUT_NEXT_GAME if i == 0 else OUT_ALL_GAMES
                absent = moved | st.isin(out_set)
                vac_f.append({"team": team, "m": m, "_ts": cand.loc[absent, "_ts"].sum(),
                              "_cs": cand.loc[absent, "_cs"].sum()})
    vac = pd.concat([vac_h.reset_index(), pd.DataFrame(vac_f)], ignore_index=True)
    vac = vac.rename(columns={"m": "k", "_ts": "vacated_tgt", "_cs": "vacated_car"})
    df = df.merge(vac, on=["team", "k"], how="left")
    df[["vacated_tgt", "vacated_car"]] = df[["vacated_tgt", "vacated_car"]].fillna(0.0)
    return df.drop(columns=["_ts", "_cs", "k"])
