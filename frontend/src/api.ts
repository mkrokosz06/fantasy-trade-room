import { useCallback, useEffect, useState } from "react";

const cache = new Map<string, unknown>();
const listeners = new Set<() => void>();

export async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, {
    ...init,
    headers: { "Content-Type": "application/json", ...(init?.headers || {}) },
  });
  const text = await res.text();
  const body = text ? JSON.parse(text) : null;
  if (!res.ok) {
    const msg = body?.detail ?? `Request failed (HTTP ${res.status})`;
    throw new Error(typeof msg === "string" ? msg : JSON.stringify(msg));
  }
  return body as T;
}

/** Clear cached GETs (after settings change or data refresh) and re-fetch mounted views. */
export function invalidateAll() {
  cache.clear();
  listeners.forEach((fn) => fn());
}

export function useApi<T>(path: string | null) {
  const [data, setData] = useState<T | undefined>(() =>
    path ? (cache.get(path) as T | undefined) : undefined,
  );
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [nonce, setNonce] = useState(0);

  useEffect(() => {
    const fn = () => setNonce((n) => n + 1);
    listeners.add(fn);
    return () => {
      listeners.delete(fn);
    };
  }, []);

  useEffect(() => {
    if (!path) return;
    let cancelled = false;
    const cached = cache.get(path) as T | undefined;
    setError(null);
    if (cached !== undefined) {
      setData(cached);
      return;
    }
    setLoading(true);
    api<T>(path)
      .then((d) => {
        cache.set(path, d);
        if (!cancelled) setData(d);
      })
      .catch((e: Error) => !cancelled && setError(e.message))
      .finally(() => !cancelled && setLoading(false));
    return () => {
      cancelled = true;
    };
  }, [path, nonce]);

  const reload = useCallback(() => {
    if (path) cache.delete(path);
    setNonce((n) => n + 1);
  }, [path]);

  return { data, error, loading, reload };
}

// ---------------------------------------------------------------- types
export type Call = "BUY LOW" | "SELL HIGH" | "HOLD";

export interface RankedPlayer {
  id: string;
  name: string;
  position: string;
  team: string | null;
  value: number;
  vorp: number;
  ros_pts: number;
  proj_ppg: number;
  next_pts: number | null;
  next_opp: string | null;
  next_week: number | null;
  pos_rank: number;
  overall_rank: number;
  games_left: number;
  sos: number;
  injury_status: string | null;
  headshot: string | null;
  age: number | null;
  floor_ppg?: number;
  ceiling_ppg?: number;
  next_floor?: number;
  next_ceiling?: number;
}

export interface Advice {
  id: string;
  name: string;
  position: string;
  team: string;
  call: Call;
  confidence: number;
  score: number;
  reasons: string[];
  proj_ppg: number;
  perceived_ppg: number;
  ppg: number;
  last3_ppg: number;
  xfp_ppg: number;
  td: number;
  xtd: number;
  value: number;
  injury_status: string | null;
  headshot: string | null;
  games: number;
  floor_ppg?: number;
  ceiling_ppg?: number;
}

export interface Meta {
  season: number;
  current_week: number;
  last_stats_week: number;
  projection_weeks: number[];
  loaded_at: string;
  warnings: string[];
  seasons: number[];
  espn_cookies_set: boolean;
  replacement: Record<string, number>;
}

export interface Settings {
  scoring: { preset: string; label: string; rules: Record<string, number>; te_rec_bonus: number };
  league: { preset: string; label: string; teams: number; slots: Record<string, number> };
  projection: { sleeper_blend: number; end_week: number; half_life_games: number };
  espn: { league_id: string; my_team_id: string };
}

export interface LeaguePlayer {
  espn_id: string;
  id: string | null;
  name: string;
  position: string | null;
  pro_team: string | null;
  slot: string;
  value?: number;
  proj_ppg?: number;
  ros_pts?: number;
  call?: Call;
}

export interface LeagueTeam {
  team_id: string;
  name: string;
  abbrev: string;
  owner: string;
  wins: number;
  losses: number;
  ties: number;
  points_for: number;
  lineup_ros_pts: number;
  roster: LeaguePlayer[];
}

export interface League {
  league_id: string;
  name: string;
  size: number;
  teams: LeagueTeam[];
  my_team_id: string | null;
  settings: {
    unmapped: { slots: number[]; stats: { stat_id: number; points: number }[] };
  };
}

export interface MetricRow {
  mae: number;
  bias: number;
  rank_corr: number;
  n: number;
}
export interface Experiment {
  name: string;
  label: string;
  description: string;
  retrain_every: number | null;
  kept?: boolean | null;
  games: number;
  metrics: Record<string, MetricRow>;
}
export interface Experiments {
  ran_at: string;
  seasons: number[];
  tune_season: number;
  rule: string;
  experiments: Experiment[];
  tuning: { name: string; rounds: number; metrics_2023: MetricRow }[];
  retrain_gain: { mae: number; rank_corr: number };
  chosen_label: string;
  seconds: number;
  final?: { original: MetricRow; chosen_model: MetricRow; blend: MetricRow; sleeper: MetricRow };
}
export interface InjuryRow {
  status: string;
  practice: string;
  n: number;
  played: number;
  multiplier: number;
}
export interface MlStatus {
  status: "idle" | "training" | "ready" | "error";
  kind?: "train" | "experiments";
  experiments: Experiments | null;
  trained: boolean;
  log: string[];
  started_at: string | null;
  finished_at: string | null;
  error: string | null;
  meta: null | {
    trained_at: string;
    seasons: [number, number];
    through_week: number;
    training_games: number;
    n_features: number;
    seconds: number;
    injury?: { base_ratio: number; by_status: Record<string, number>; rows: InjuryRow[] };
    backtest: {
      seasons: number[];
      games: number;
      blend_sleeper_weight: number;
      blend_by_pos?: Record<string, number>;
      ensemble_alpha?: number;
      range_coverage: number;
      range_coverage_raw?: number;
      range_scale?: number;
      range_width: number;
      methods: ({ key: string; label: string; all: MetricRow } & Record<string, MetricRow | string>)[];
    };
    importance: Record<string, { feature: string; label: string; share: number }[]>;
  };
}

export interface TeamOdds {
  team_id: string;
  team_name: string;
  situation: string;
  playoff_before: number;
  playoff_after: number;
  title_before: number;
  title_after: number;
  bye_before: number;
  bye_after: number;
  avg_wins_before: number;
  avg_wins_after: number;
  playoff_delta: number;
  title_delta: number;
  must_drop?: { id: string; name: string }[];
}
export interface Acceptance {
  probability: number;
  appeal: number;
  label: "Strong offer" | "Fair offer" | "Long shot";
  reasons: string[];
  perceived_get: number;
  perceived_give: number;
  league_accept_rate: { proposals: number; completed: number };
}
export interface Standing {
  team_id: string;
  name: string;
  owner: string;
  wins: number;
  losses: number;
  ties: number;
  points_for: number;
  playoff: number;
  bye: number;
  title: number;
  avg_wins: number;
  avg_seed: number;
  situation: string;
  activity: { proposed: number; received: number; declined: number; completed: number };
}
export interface LeagueOdds {
  engine: string;
  standings: Standing[];
  playoffs: { regular_season_weeks: number; teams: number };
  sims: number;
  league_accept_rate: { proposals: number; completed: number };
}

export const odds = (p: number | null | undefined) =>
  p === null || p === undefined ? "–" : p >= 0.995 ? ">99%" : p > 0 && p < 0.005 ? "<1%" : `${Math.round(p * 100)}%`;

// ---------------------------------------------------------------- formatting
export const fmt = (n: number | null | undefined, d = 1) =>
  n === null || n === undefined || Number.isNaN(n) ? "–" : n.toFixed(d);
export const pct = (n: number | null | undefined, d = 0) =>
  n === null || n === undefined || Number.isNaN(n) ? "–" : `${(n * 100).toFixed(d)}%`;
