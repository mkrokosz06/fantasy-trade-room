# Trade Room: fantasy football trade calculator, stats and projections

A local web app for redraft fantasy football. It evaluates trades, tells you who to buy low, sell high or hold,
projects every player for each remaining week, and connects to your ESPN league. It uses only free data sources.

## What it does

| Page | What you get |
| --- | --- |
| **Trade calculator** | Add players to each side and see who wins, drawn on a football field. Pick ESPN teams to also see how each team's projected starting lineup changes. Suggests players that would even out an uneven trade. |
| **Trade advisor** | Buy low / sell high / hold calls for every fantasy-relevant player, each with reasons. Includes touchdown luck, points vs expected points, usage trend, remaining schedule and injuries. With your ESPN team selected, it also searches every roster for trades that improve your lineup. |
| **Rankings** | Rest-of-season values, or projections for any upcoming week, with matchup, Vegas and injury adjustments. |
| **Player stats** | Season tables scored with your settings, including expected fantasy points (xFP), expected TDs, target share and WOPR. Click any player for charts, game logs and a breakdown of how the projection was built. |
| **Matchups** | Fantasy points each defense allows to each position. |
| **ESPN league** | Rosters, standings by projected lineup strength, and a one-click import of your league's scoring and lineup slots. |
| **Settings** | Scoring presets (PPR is the default; Half-PPR, Standard, 6-pt pass TD and TE premium are also available), league size and lineup slots (1QB, 3WR, Superflex, 2QB), and projection controls. |

## Running it

Requirements: Python 3.12+ and Node 20+.

```powershell
./start.ps1
```

The first run installs everything. After that it opens http://localhost:5173. To start things by hand:

```powershell
# backend
python -m venv .venv
.venv\Scripts\pip install -r backend\requirements.txt
cd backend; ..\.venv\Scripts\python -m uvicorn app.main:app --port 8000 --reload

# frontend (second terminal)
cd frontend; npm install; npx vite
```

Tests: `cd backend; ..\.venv\Scripts\python -m pytest`

## Connecting ESPN

1. Open your league on ESPN and copy the number after `leagueId=` in the URL.
2. Enter it on the **ESPN league** page in the app, then choose your team.
3. **Private leagues** also need two login cookies. Copy `.env.example` to `.env`, then:
   log in at fantasy.espn.com, press F12, go to *Application > Cookies > https://fantasy.espn.com*,
   and copy the values of `espn_s2` and `SWID` into `.env`. Restart the backend.
   `.env` is gitignored. Never commit these cookies; they work like your ESPN password.
4. Click **Use this league's scoring and lineup** to import your league's settings. Any scoring rules the app
   doesn't model yet (yardage bonuses, D/ST) are listed so you know what's missing.

## How the numbers work

**Data sources (all free, no API keys):**
- [nflverse](https://github.com/nflverse/nflverse-data): weekly player stats, schedules, Vegas lines. Updated nightly.
- [Sleeper API](https://docs.sleeper.com): current week, injury statuses, per-stat weekly projections.
- [DynastyProcess](https://github.com/dynastyprocess/data): player ID crosswalk between nflverse, Sleeper and ESPN.
- ESPN's unofficial fantasy API for your league.

Everything is cached in `data/cache/` and refreshed every 12 hours. Use **Settings > Refresh data now** to refresh sooner.

**Projections (v1, `backend/app/projections.py`):**
1. *Usage*: recency-weighted pass attempts, carries and targets per game (half-life of 5 games; prior seasons discounted).
2. *Efficiency*: yards and TDs per opportunity, shrunk toward the position average until the sample is large
   enough. This is what makes TD spikes regress.
3. *Context*: an opponent factor (points allowed to the position, shrunk early in the season) and a Vegas factor
   (implied team total vs that team's average) scale yardage and TDs.
4. *Blend*: mixed with Sleeper's projections (35% by default, more for rookies and thin samples), then multiplied by
   the chance of playing based on injury status.
5. *Score*: converted to points with your scoring settings.

**Value** is rest-of-season projected points minus a replacement-level player at the same position. Replacement
level comes from your league size, starting slots and part of the bench, so Superflex raises QB values automatically.

**Trades** add up the values on each side with a consolidation adjustment: one great player beats two average ones.
With ESPN rosters, the app also projects each team's best possible starting lineup for every remaining week,
before and after the trade.

**Buy low / sell high** compares *perceived* points per game (half season average, half last 3 games, which is
roughly how trade partners judge players) with *projected* points per game. The comparison is adjusted for
position-wide bias and nudged by usage trend. Confidence is lower for small samples, long injuries, and when the
usage trend argues against the call.

## Project layout

```
backend/app/
  data/        sources.py (nflverse, Sleeper, ID crosswalk), espn.py, store.py (joins it all), cache.py
  scoring.py   canonical stats, source mappings, scoring presets
  projections.py, values.py, trade.py, advisor.py, stats.py
  main.py      FastAPI endpoints under /api
backend/tests/ unit tests
frontend/src/  React + TypeScript (Vite); pages/ holds one file per page
```

## Roadmap

- Trained projection model on historical nflverse data, back-tested against this v1 baseline
- D/ST and yardage-bonus scoring
- Snap counts and red-zone usage in the model
- Yahoo league support
