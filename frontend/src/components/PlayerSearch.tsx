import { useId, useMemo, useState } from "react";
import { fmt } from "../api";
import { Headshot } from "./bits";

export interface SearchItem {
  id: string;
  name: string;
  position: string | null;
  team: string | null;
  value?: number;
  headshot?: string | null;
}

export function PlayerSearch({
  items,
  exclude,
  onPick,
  placeholder = "Add a player",
}: {
  items: SearchItem[];
  exclude: Set<string>;
  onPick: (p: SearchItem) => void;
  placeholder?: string;
}) {
  const [q, setQ] = useState("");
  const [open, setOpen] = useState(false);
  const [hl, setHl] = useState(0);
  const listId = useId();

  const matches = useMemo(() => {
    const s = q.trim().toLowerCase();
    const pool = items.filter((p) => !exclude.has(p.id));
    if (!s) return pool.slice(0, 12);
    return pool.filter((p) => p.name.toLowerCase().includes(s)).slice(0, 12);
  }, [q, items, exclude]);

  const pick = (p: SearchItem) => {
    onPick(p);
    setQ("");
    setHl(0);
  };

  return (
    <div className="search">
      <input
        className="input"
        value={q}
        placeholder={placeholder}
        role="combobox"
        aria-expanded={open}
        aria-controls={listId}
        onChange={(e) => {
          setQ(e.target.value);
          setOpen(true);
          setHl(0);
        }}
        onFocus={() => setOpen(true)}
        onBlur={() => setTimeout(() => setOpen(false), 150)}
        onKeyDown={(e) => {
          if (e.key === "ArrowDown") setHl((h) => Math.min(h + 1, matches.length - 1));
          else if (e.key === "ArrowUp") setHl((h) => Math.max(h - 1, 0));
          else if (e.key === "Enter" && matches[hl]) pick(matches[hl]);
          else if (e.key === "Escape") setOpen(false);
        }}
      />
      {open && matches.length > 0 && (
        <div className="results" id={listId} role="listbox">
          {matches.map((p, i) => (
            <button
              key={p.id}
              role="option"
              aria-selected={i === hl}
              className={i === hl ? "hl" : ""}
              onMouseDown={(e) => e.preventDefault()}
              onClick={() => pick(p)}
            >
              <Headshot src={p.headshot} size={24} />
              <span className="grow">{p.name}</span>
              <span className="pos">
                {p.position} {p.team ?? "FA"}
              </span>
              {p.value !== undefined && <span className="num small muted">{fmt(p.value, 0)}</span>}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
