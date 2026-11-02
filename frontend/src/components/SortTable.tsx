import { useMemo, useState, type ReactNode } from "react";

export interface Col<T> {
  key: string;
  label: string;
  num?: boolean;
  sort?: (r: T) => number | string | null | undefined;
  render: (r: T) => ReactNode;
  title?: string;
}

export function SortTable<T>({
  rows,
  cols,
  initial,
  rowKey,
  limit,
}: {
  rows: T[];
  cols: Col<T>[];
  initial?: { key: string; desc: boolean };
  rowKey: (r: T) => string;
  limit?: number;
}) {
  const [sort, setSort] = useState(initial ?? { key: cols[0].key, desc: false });
  const sorted = useMemo(() => {
    const col = cols.find((c) => c.key === sort.key);
    if (!col?.sort) return rows;
    const get = col.sort;
    return [...rows].sort((a, b) => {
      const va = get(a);
      const vb = get(b);
      if (va === vb) return 0;
      if (va === null || va === undefined) return 1;
      if (vb === null || vb === undefined) return -1;
      const r = va < vb ? -1 : 1;
      return sort.desc ? -r : r;
    });
  }, [rows, cols, sort]);
  const shown = limit ? sorted.slice(0, limit) : sorted;

  return (
    <div className="tablewrap">
      <table>
        <thead>
          <tr>
            {cols.map((c) => (
              <th
                key={c.key}
                className={`${c.num ? "n" : ""} ${c.sort ? "sortable" : ""}`}
                title={c.title}
                aria-sort={sort.key === c.key ? (sort.desc ? "descending" : "ascending") : undefined}
                onClick={() =>
                  c.sort &&
                  setSort((s) => ({ key: c.key, desc: s.key === c.key ? !s.desc : !!c.num }))
                }
              >
                {c.label}
                {sort.key === c.key ? (sort.desc ? " ▾" : " ▴") : ""}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {shown.map((r) => (
            <tr key={rowKey(r)}>
              {cols.map((c) => (
                <td key={c.key} className={c.num ? "n" : ""}>
                  {c.render(r)}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
