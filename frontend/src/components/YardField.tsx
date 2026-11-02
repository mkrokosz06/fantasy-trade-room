/**
 * The trade verdict drawn as a football field. Side A's end zone is on the
 * left, side B's on the right. The winning side drives the ball toward the
 * other side's end zone; a fair trade sits at the 50.
 */
export function YardField({
  aLabel,
  bLabel,
  aGets,
  bGets,
}: {
  aLabel: string;
  bLabel: string;
  aGets: number;
  bGets: number;
}) {
  const total = aGets + bGets;
  // Share of value going to A. More to A = A drives toward B's end zone (right).
  const share = total > 0 ? aGets / total : 0.5;
  const left = 9 + share * 82; // keep the ball between the end zones
  const yards = [10, 20, 30, 40, 50, 40, 30, 20, 10];
  return (
    <div
      className="field"
      role="img"
      aria-label={`Trade balance: ${aLabel} receives ${aGets.toFixed(1)} value, ${bLabel} receives ${bGets.toFixed(1)}`}
    >
      <div className="endzone">{aLabel}</div>
      {yards.map((y, i) => (
        <div key={i} className="yard" style={{ left: `${9 + (i + 1) * 8.2}%` }}>
          <span>{y}</span>
        </div>
      ))}
      <div className="flag" style={{ left: "50%" }} title="Midfield = fair" />
      <div className="ball" style={{ left: `${left}%` }} />
      <div className="endzone right">{bLabel}</div>
    </div>
  );
}
