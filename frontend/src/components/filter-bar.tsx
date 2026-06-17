import { Filter, X } from "lucide-react";

export type Decade = "all" | "1990s" | "2000s" | "2010s" | "2020s";
export type SortBy = "title" | "year-desc" | "year-asc";

interface FilterBarProps {
  decade: Decade;
  onDecadeChange: (d: Decade) => void;
  sortBy: SortBy;
  onSortChange: (s: SortBy) => void;
  minScore: number;
  onMinScoreChange: (v: number) => void;
  onReset: () => void;
  hasActive: boolean;
}

const SELECT_CLS =
  "h-9 rounded-lg bg-secondary/60 border border-border/60 px-3 text-sm focus:outline-none focus:ring-2 focus:ring-primary/40";

export function FilterBar({
  decade,
  onDecadeChange,
  sortBy,
  onSortChange,
  minScore,
  onMinScoreChange,
  onReset,
  hasActive,
}: FilterBarProps) {
  return (
    <div className="mb-6 rounded-2xl border border-border/60 bg-card/40 backdrop-blur-sm p-4">
      <div className="flex items-center gap-3 flex-wrap">
        <div className="inline-flex items-center gap-2 text-sm font-medium text-muted-foreground">
          <Filter className="h-4 w-4" />
          Filters
        </div>

        <label className="flex items-center gap-2 text-xs text-muted-foreground">
          Decade
          <select
            value={decade}
            onChange={(e) => onDecadeChange(e.target.value as Decade)}
            className={SELECT_CLS}
          >
            <option value="all">All</option>
            <option value="1990s">1990s</option>
            <option value="2000s">2000s</option>
            <option value="2010s">2010s</option>
            <option value="2020s">2020s</option>
          </select>
        </label>

        <label className="flex items-center gap-2 text-xs text-muted-foreground">
          Sort
          <select
            value={sortBy}
            onChange={(e) => onSortChange(e.target.value as SortBy)}
            className={SELECT_CLS}
          >
            <option value="title">Title (A–Z)</option>
            <option value="year-desc">Year (new → old)</option>
            <option value="year-asc">Year (old → new)</option>
          </select>
        </label>

        <label className="flex items-center gap-3 text-xs text-muted-foreground min-w-[200px]">
          Min match
          <input
            type="range"
            min={0}
            max={100}
            value={minScore}
            onChange={(e) => onMinScoreChange(Number(e.target.value))}
            className="flex-1 accent-primary"
          />
          <span className="tabular-nums w-9 text-right text-foreground font-medium">
            {minScore}%
          </span>
        </label>

        {hasActive && (
          <button
            onClick={onReset}
            className="ml-auto inline-flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground transition-colors"
          >
            <X className="h-3 w-3" />
            Reset
          </button>
        )}
      </div>
    </div>
  );
}
