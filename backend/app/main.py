"""FastAPI app: run with  uvicorn app.main:app --reload  from backend/."""
from __future__ import annotations

import json
import threading
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from . import advisor, config, stats, tracking, trade
from .data import espn
from .data.cache import clear_cache
from .data.store import get_store
from .league import finder
from .league.context import LeagueContext
from .league.sim import situation
from .ml import model as ml_model
from .ml import service as ml_service
from .projections import RATIOS, get_projections
from .scoring import SCORING_PRESETS, STAT_LABELS, score_dst
from .settings_store import (LEAGUE_PRESETS, SLOT_LABELS, load_settings, save_settings,
                             settings_key)
from .values import LineupEvaluator, player_values

app = FastAPI(title="Fantasy Football Toolkit")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
                   allow_methods=["*"], allow_headers=["*"])


@app.on_event("startup")
def _warm_up():
    """Build projections, model predictions and league odds in the background at
    startup, so the first page load isn't the slow one."""
    def run():
        for engine in ("v1", "ml"):
            try:
                c = ctx(engine)
                _ = c.signals
                if c.settings["espn"].get("league_id") or config.ESPN_LEAGUE_ID:
                    league_ctx(c)
            except Exception:  # noqa: BLE001 - warm-up is best effort (e.g. model not trained)
                continue
    threading.Thread(target=run, daemon=True, name="warm-up").start()


def records(df: pd.DataFrame) -> list[dict]:
    if df is None or df.empty:
        return []
    return json.loads(df.to_json(orient="records", date_format="iso"))


# ---------------------------------------------------------------- shared context
ENGINES = ("v1", "ml")


class Ctx:
    """Everything an endpoint needs, for one projection engine:
    v1 = usage model + Sleeper; ml = trained LightGBM model + Sleeper."""

    def __init__(self, engine: str = "v1"):
        self.engine = engine
        self.store = get_store()
        self.settings = load_settings()
        self.model_meta = None
        model = model_key = None
        if engine == "ml":
            end_week = int(self.settings["projection"].get("end_week", 17))
            preds, meta = ml_service.predictions(self.store, end_week)
            bt = meta["backtest"]
            model = (preds, bt.get("blend_by_pos") or float(bt["blend_sleeper_weight"]))
            model_key = meta["trained_at"]
            self.model_meta = meta
        self.proj = get_projections(self.store, self.settings, model, model_key)
        self.values, self.replacement = player_values(self.proj.summary, self.settings["league"])
        self._season_tbl: dict[int, pd.DataFrame] = {}
        self._signals = None
        self._evaluator = None

    def season_table(self, season: int) -> pd.DataFrame:
        if season not in self._season_tbl:
            self._season_tbl[season] = stats.season_table(
                self.store, self.proj, self.settings["scoring"], season)
        return self._season_tbl[season]

    @property
    def signals(self) -> pd.DataFrame:
        if self._signals is None:
            self._signals = advisor.signals(self.season_table(self.store.season), self.values,
                                            self.store.players)
            try:  # save this week's calls once, to grade them later
                tracking.snapshot(self._signals, self.store.season, self.store.last_stats_week, self.engine)
            except OSError:
                pass
        return self._signals

    @property
    def evaluator(self) -> LineupEvaluator:
        if self._evaluator is None:
            self._evaluator = LineupEvaluator(self.proj.weekly, self.settings["league"]["slots"])
        return self._evaluator


_ctx: dict[str, tuple[tuple, Ctx]] = {}
_ctx_lock = threading.Lock()


def ctx(engine: str = "v1") -> Ctx:
    if engine not in ENGINES:
        raise HTTPException(400, f"Unknown engine '{engine}'. Use one of {ENGINES}.")
    with _ctx_lock:
        store = get_store()
        trained = (ml_model.load_meta() or {}).get("trained_at") if engine == "ml" else None
        key = (store.loaded_at, settings_key(load_settings()), trained)
        cached = _ctx.get(engine)
        if cached is None or cached[0] != key:
            try:
                _ctx[engine] = (key, Ctx(engine))
            except ml_service.NotTrained as e:
                raise HTTPException(409, str(e)) from e
        return _ctx[engine][1]


# ---------------------------------------------------------------- meta & settings
@app.get("/api/meta")
def meta():
    c = ctx()
    return {
        "season": c.store.season, "current_week": c.store.current_week,
        "last_stats_week": c.store.last_stats_week, "projection_weeks": c.proj.weeks,
        "loaded_at": c.store.loaded_at.isoformat(), "warnings": c.store.warnings,
        "seasons": sorted(c.store.weekly["season"].unique().tolist(), reverse=True),
        "espn_cookies_set": bool(config.ESPN_S2 and config.ESPN_SWID),
        "replacement": c.replacement,
    }


@app.get("/api/presets")
def presets():
    return {"scoring": SCORING_PRESETS, "league": LEAGUE_PRESETS,
            "slot_labels": SLOT_LABELS, "stat_labels": STAT_LABELS}


@app.get("/api/settings")
def get_settings():
    return load_settings()


@app.put("/api/settings")
def put_settings(body: dict[str, Any]):
    return save_settings(body)


@app.post("/api/refresh")
def refresh():
    n = clear_cache(keep_immutable=True)
    get_store(force=True)
    return {"cleared_files": n, "meta": meta()}


# ---------------------------------------------------------------- players & stats
@app.get("/api/players")
def players(pos: str | None = None, q: str | None = None, season: int | None = None,
            limit: int = Query(300, le=2000), engine: str = "v1"):
    c = ctx(engine)
    season = season or c.store.season
    tbl = c.season_table(season)
    v = c.values[["value", "vorp", "ros_pts", "proj_ppg", "next_pts", "next_opp", "pos_rank",
                  "overall_rank", "sos", "games_left"]]
    base = c.store.players[["id", "name", "position", "team", "age", "injury_status", "headshot"]]
    df = base.join(v, how="left").join(
        tbl.drop(columns=["name", "position", "team", "pos_rank"], errors="ignore"), how="left")
    df = df[df["value"].notna() | df["games"].notna()]
    if pos:
        df = df[df["position"].isin(pos.upper().split(","))]
    if q:
        df = df[df["name"].str.contains(q, case=False, na=False, regex=False)]
    if c.store.season == season:
        df = df.sort_values("value", ascending=False, na_position="last")
    else:
        df = df.sort_values("pts", ascending=False, na_position="last")
    return records(df.head(limit))


@app.get("/api/players/{pid}")
def player(pid: str, engine: str = "v1"):
    c = ctx(engine)
    if pid not in c.store.players.index:
        raise HTTPException(404, "Player not found")
    info = c.store.players.loc[pid]
    out: dict[str, Any] = {"info": json.loads(info.to_json())}
    if pid in c.values.index:
        out["value"] = json.loads(c.values.loc[pid].to_json())
    if info["position"] == "DST":
        dw = c.store.dst_weekly
        g = dw[dw["id"] == pid].copy() if len(dw) else pd.DataFrame()
        if len(g):
            g["pts"] = score_dst(g, c.settings["scoring"].get("dst"))
            g = g.sort_values(["season", "week"], ascending=False)
        out["game_log"] = records(g)
    else:
        out["game_log"] = records(stats.game_log(c.store, c.proj, c.settings["scoring"], pid))
    seasons = []
    for s in sorted(c.store.weekly["season"].unique(), reverse=True):
        t = c.season_table(int(s))
        if pid in t.index:
            seasons.append({"season": int(s), **json.loads(t.loc[pid].to_json())})
    out["seasons"] = seasons
    wk = c.proj.weekly[c.proj.weekly["id"] == pid]
    pcols = ["week", "opp", "is_home", "gameday", "matchup", "vegas", "avail", "sleeper_weight",
             "pts", "floor", "ceiling", "pass_yd", "pass_td", "rush_att", "rush_yd", "rush_td",
             "rec_tgt", "rec", "rec_yd", "rec_td"]
    out["projection"] = records(wk[[col for col in pcols if col in wk.columns]])
    out["engine"] = c.engine
    if pid in c.proj.base.index:
        b = c.proj.base.loc[pid]
        pr = c.proj.priors.loc[b["position"]] if b["position"] in c.proj.priors.index else None
        out["model"] = {
            "weighted_games": round(float(b["W"]), 2),
            "per_game": {s: round(float(b[s]), 2) for s in ["pass_att", "rush_att", "rec_tgt"]},
            "efficiency": [{"stat": s, "per": d, "player": round(float(b[s + "_eff"]), 4),
                            "position_avg": round(float(pr[s]), 4) if pr is not None else None}
                           for s, d, _ in RATIOS],
        }
    if pid in c.signals.index:
        sg = c.signals.loc[pid]
        out["signal"] = {"call": sg["call"], "reasons": sg["reasons"], "edge": float(sg["edge"]),
                         "confidence": float(sg["confidence"])}
    return out


@app.get("/api/rankings")
def rankings(pos: str | None = None, limit: int = Query(300, le=2000), engine: str = "v1"):
    c = ctx(engine)
    df = c.values.join(c.store.players[["injury_status", "headshot", "age"]], how="left")
    if pos:
        df = df[df["position"].isin(pos.upper().split(","))]
    return records(df.sort_values("value", ascending=False).head(limit).reset_index())


@app.get("/api/projections/week/{week}")
def week_projections(week: int, pos: str | None = None, limit: int = Query(300, le=2000),
                     engine: str = "v1"):
    c = ctx(engine)
    wk = c.proj.weekly[c.proj.weekly["week"] == week]
    if pos:
        wk = wk[wk["position"].isin(pos.upper().split(","))]
    wk = wk.sort_values("pts", ascending=False).head(limit)
    wk = wk.join(c.store.players[["headshot"]], on="id")
    return records(wk)


@app.get("/api/matchups")
def matchups():
    c = ctx()
    return {"defense": records(stats.defense_table(c.proj)),
            "schedule": records(c.proj.weekly[["team", "week", "opp", "is_home"]]
                                .drop_duplicates(["team", "week"]))}


# ---------------------------------------------------------------- trades & advice
class TradeIn(BaseModel):
    side_a: list[str]
    side_b: list[str]
    team_a: str | None = None
    team_b: str | None = None


@app.post("/api/trade")
def evaluate_trade(body: TradeIn, engine: str = "v1"):
    c = ctx(engine)
    L = _league_ctx_or_none(c)
    team_a, team_b = body.team_a, body.team_b
    if L is not None:
        # Work out whose players these are if teams weren't picked.
        team_a = team_a or L.team_of(body.side_a)
        team_b = team_b or L.team_of(body.side_b)
    roster_a = roster_b = None
    if L is not None and team_a and team_b and team_a != team_b:
        roster_a, roster_b = L.rosters.get(team_a), L.rosters.get(team_b)
    res = trade.evaluate(c.values, body.side_a, body.side_b, c.evaluator if roster_a else None,
                         roster_a, roster_b)
    for side in ("side_a", "side_b"):
        for p in res[side]["gives"]:
            if p["id"] in c.signals.index:
                p["call"] = c.signals.at[p["id"], "call"]
    if roster_a is not None and roster_b is not None:
        odds = L.trade_odds(team_a, team_b, body.side_a, body.side_b)
        res["odds"] = odds
        # Assume side A is proposing: would side B accept?
        res["acceptance"] = L.acceptance(team_b, they_get=body.side_a, they_give=body.side_b, odds=odds["b"])
        res["teams"] = {"a": team_a, "b": team_b}
    return res


@app.get("/api/advice")
def advice(pos: str | None = None, call: str | None = None, engine: str = "v1"):
    c = ctx(engine)
    df = c.signals
    if df.empty:
        return []
    if pos:
        df = df[df["position"].isin(pos.upper().split(","))]
    if call:
        df = df[df["call"] == call]
    df = df.join(c.store.players[["headshot"]], how="left")
    cols = ["id", "name", "position", "team", "call", "confidence", "edge", "score", "reasons",
            "proj_ppg", "perceived_ppg", "ppg", "last3_ppg", "xfp_ppg", "perf_gap", "td", "xtd",
            "td_gap", "usage_trend", "sos", "sos_rank", "value", "perceived_value",
            "injury_status", "headshot", "games"]
    cols += [col for col in ("floor_ppg", "ceiling_ppg") if col in df.columns]
    out = df.reset_index()[cols]
    # What leaguemates see: ESPN's own rest-of-season projection per game.
    L = _league_ctx_or_none(c)
    if L is not None and not L.espn.empty and "espn_ros_avg" in L.espn.columns:
        out["espn_ppg"] = out["id"].map(L.espn["espn_ros_avg"]).round(2)
    return records(out)


@app.get("/api/advice/backtest")
def advice_backtest():
    c = ctx("v1")
    return tracking.backtest_advisor(c.store, c.settings["scoring"])


@app.get("/api/advice/track")
def advice_track(engine: str = "v1"):
    c = ctx(engine)
    _ = c.signals  # make sure this week's snapshot exists
    return tracking.grade(c.store, c.settings["scoring"], engine)


# ---------------------------------------------------------------- ML model
@app.get("/api/ml/status")
def ml_status():
    return ml_service.status()


@app.post("/api/ml/train")
def ml_train(kind: str = "train"):
    if kind not in ("train", "experiments"):
        raise HTTPException(400, "kind must be 'train' or 'experiments'")
    started = ml_service.start_training(kind)
    return {"started": started, "status": ml_service.status()}


# ---------------------------------------------------------------- ESPN league
def _league(c: Ctx, force: bool = False) -> dict:
    league_id = c.settings["espn"].get("league_id") or config.ESPN_LEAGUE_ID
    espn_map = c.store.players.dropna(subset=["espn_id"]).set_index("espn_id")["id"].to_dict()
    try:
        raw = espn.fetch_league(league_id, c.store.season, force=force)
    except espn.ESPNError as e:
        raise HTTPException(400, str(e)) from e
    return espn.parse_league(raw, espn_map)


def _rosters(lg: dict) -> dict[str, list[str]]:
    return {t["team_id"]: [p["id"] for p in t["roster"] if p["id"]] for t in lg["teams"]}


_lctx: dict[str, tuple] = {}
_lctx_lock = threading.Lock()


def league_ctx(c: Ctx) -> LeagueContext:
    """League standings/odds/acceptance context, rebuilt when projections or rosters change."""
    lg = _league(c)
    league_id = c.settings["espn"].get("league_id") or config.ESPN_LEAGUE_ID
    key = (id(c), tuple(sorted((k, tuple(v)) for k, v in _rosters(lg).items())),
           tuple((g["period"], g["winner"]) for g in lg["schedule"]))
    with _lctx_lock:
        cached = _lctx.get(c.engine)
        if cached is None or cached[0] != key:
            _lctx[c.engine] = (key, LeagueContext(c, lg, league_id))
        return _lctx[c.engine][1]


def _league_ctx_or_none(c: Ctx) -> LeagueContext | None:
    if not (c.settings["espn"].get("league_id") or config.ESPN_LEAGUE_ID):
        return None
    try:
        return league_ctx(c)
    except HTTPException:
        return None


def _best_engine() -> str:
    """League-wide odds use the trained model when it exists (it has weekly ranges)."""
    return "ml" if ml_model.load_meta() else "v1"


def _player_info(c: Ctx, pid: str) -> dict:
    r = c.values.loc[pid] if pid in c.values.index else None
    return {"id": pid, "name": r["name"] if r is not None else pid,
            "position": r["position"] if r is not None else None,
            "team": r["team"] if r is not None else None,
            "value": float(r["value"]) if r is not None else 0.0,
            "call": c.signals.at[pid, "call"] if pid in c.signals.index else None}


@app.get("/api/league/odds")
def league_odds(engine: str | None = None):
    c = ctx(engine or _best_engine())
    L = league_ctx(c)
    return {"engine": c.engine, "standings": L.standings(), "playoffs": L.league["playoffs"],
            "sims": int(L.sim.z.shape[0]), "weeks": L.sim.weeks,
            "league_accept_rate": {"proposals": sum(a.get("proposed", 0) for a in L.activity.values()),
                                   "completed": len(L.history)}}


@app.get("/api/league/waivers")
def league_waivers(team_id: str | None = None, engine: str = "v1"):
    c = ctx(engine)
    team_id = team_id or c.settings["espn"].get("my_team_id")
    if not team_id:
        raise HTTPException(400, "Pick your team on the League page first.")
    L = league_ctx(c)
    if team_id not in L.rosters:
        raise HTTPException(404, f"Team {team_id} not in league.")
    opts = L.waiver_options(team_id, c.evaluator)
    for o in opts:
        o["add"] = {**_player_info(c, o["add"]), "headshot": c.store.players["headshot"].get(o["add"]),
                    "proj_ppg": float(c.values["proj_ppg"].get(o["add"], 0.0))}
        o["drop"] = {**_player_info(c, o["drop"]), "proj_ppg": float(c.values["proj_ppg"].get(o["drop"], 0.0))}
    return {"team_id": team_id, "options": opts, "free_agents_checked": len(L.free_agents())}


@app.get("/api/league/lineup")
def league_lineup(team_id: str | None = None, engine: str = "v1"):
    c = ctx(engine)
    team_id = team_id or c.settings["espn"].get("my_team_id")
    if not team_id:
        raise HTTPException(400, "Pick your team on the League page first.")
    L = league_ctx(c)
    if team_id not in L.rosters:
        raise HTTPException(404, f"Team {team_id} not in league.")
    return L.lineup_advice(team_id)


@app.get("/api/league/history")
def league_history(engine: str | None = None):
    c = ctx(engine or _best_engine())
    return {"trades": league_ctx(c).trade_history()}


@app.get("/api/league")
def league(refresh: bool = False, engine: str = "v1"):
    c = ctx(engine)
    lg = _league(c, force=refresh)
    vals = c.values
    sig = c.signals
    for t in lg["teams"]:
        ids = [p["id"] for p in t["roster"] if p["id"]]
        t["lineup_ros_pts"] = c.evaluator.total(ids)
        for p in t["roster"]:
            pid = p["id"]
            if pid and pid in vals.index:
                p.update(value=float(vals.at[pid, "value"]), proj_ppg=float(vals.at[pid, "proj_ppg"]),
                         ros_pts=float(vals.at[pid, "ros_pts"]))
            if pid and pid in sig.index:
                p["call"] = sig.at[pid, "call"]
        t["roster"].sort(key=lambda p: -(p.get("value") or 0))
    lg["my_team_id"] = c.settings["espn"].get("my_team_id") or None
    return lg


@app.post("/api/league/import-settings")
def import_league_settings():
    c = ctx()
    lg = _league(c, force=True)
    conv = lg["settings"]
    s = load_settings()
    s["league"] = conv["league"]
    s["scoring"] = conv["scoring"]
    save_settings(s)
    return {"imported": conv, "settings": load_settings()}


@app.get("/api/league/trades")
def league_trades(team_id: str | None = None, engine: str = "v1"):
    c = ctx(engine)
    team_id = team_id or c.settings["espn"].get("my_team_id")
    if not team_id:
        raise HTTPException(400, "Pick your team on the League page first.")
    L = league_ctx(c)
    if team_id not in L.rosters:
        raise HTTPException(404, f"Team {team_id} not in league.")
    found = finder.find(L, team_id, c.evaluator)
    for f in found:
        f["give"] = [_player_info(c, p) for p in f["give"]]
        f["get"] = [_player_info(c, p) for p in f["get"]]
    me = L.odds[team_id]
    return {"team_id": team_id, "trades": found,
            "my_odds": {"playoff": me["playoff"], "title": me["title"], "situation": situation(me["playoff"])}}
