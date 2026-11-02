import { useMemo, useState } from "react";
import { fmt, useApi } from "../api";
import { ErrorBox, Loading } from "../components/bits";

interface Def {
  opp: string;
  position: string;
  allowed: number;
  games: number;
  factor: number;
  rank: number;
}

const POS = ["QB", "RB", "WR", "TE", "K"];

/** Tint a cell: easier matchups (factor > 1) lean green, tougher lean red. */
function tint(f: number) {
  const strength = Math.min(Math.abs(f - 1) / 0.2, 1) * 38;
  const color = f >= 1 ? "var(--good)" : "var(--bad)";
  return `color-mix(in srgb, ${color} ${strength.toFixed(0)}%, transparent)`;
}

export default function Matchups() {
  const m = useApi<{ defense: Def[] }>("/api/matchups");
  const [sortPos, setSortPos] = useState("RB");

  const teams = useMemo(() => {
    const by = new Map<string, Record<string, Def>>();
    m.data?.defense.forEach((d) => {
      if (!by.has(d.opp)) by.set(d.opp, {});
      by.get(d.opp)![d.position] = d;
    });
    return [...by.entries()].sort((a, b) => (a[1][sortPos]?.rank ?? 99) - (b[1][sortPos]?.rank ?? 99));
  }, [m.data, sortPos]);

  return (
    <div className="stack">
      <div className="page-head">
        <div>
          <h1>Defense vs position</h1>
          <p>
            Fantasy points each defense allows per game, by position, this season. Rank 1 is the easiest matchup. Green
            cells are easier than league average, red cells are tougher. Early-season numbers are pulled toward average.
          </p>
        </div>
      </div>
      <ErrorBox error={m.error} />
      {!m.data ? (
        <Loading />
      ) : (
        <div className="tablewrap heat">
          <table>
            <thead>
              <tr>
                <th>Defense</th>
                {POS.map((p) => (
                  <th key={p} className="n sortable" onClick={() => setSortPos(p)} aria-sort={sortPos === p ? "ascending" : undefined}>
                    {p} {sortPos === p ? "▴" : ""}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {teams.map(([team, row]) => (
                <tr key={team}>
                  <td><b>{team}</b></td>
                  {POS.map((p) => {
                    const d = row[p];
                    return (
                      <td key={p} className="h" style={{ background: d ? tint(d.factor) : undefined }} title={d ? `${fmt(d.allowed)} pts/game allowed over ${d.games} games, factor ${fmt(d.factor, 2)}` : ""}>
                        {d ? (
                          <>
                            {fmt(d.allowed)} <span className="muted small">#{d.rank}</span>
                          </>
                        ) : (
                          "–"
                        )}
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
