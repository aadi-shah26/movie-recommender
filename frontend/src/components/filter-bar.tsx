import { Filter, X } from "lucide-react";

export type SortBy = "title" | "rating-desc" | "votes-desc";
export type RuntimeBucket = "all" | "short" | "medium" | "long" | "epic";
export type MinRating = "all" | "7.5" | "8" | "8.5" | "9";

interface FilterBarProps {
  selectedGenres: string[];
  onGenresChange: (genres: string[]) => void;
  availableGenres: string[];
  minRating: MinRating;
  onMinRatingChange: (v: MinRating) => void;
  certificate: string;
  onCertificateChange: (v: string) => void;
  availableCertificates: string[];
  runtime: RuntimeBucket;
  onRuntimeChange: (v: RuntimeBucket) => void;
  director: string;
  onDirectorChange: (v: string) => void;
  availableDirectors: string[];
  sortBy: SortBy;
  onSortChange: (s: SortBy) => void;
  onReset: () => void;
  hasActive: boolean;
}

const SELECT_CLS =
  "h-9 rounded-lg bg-secondary/60 border border-border/60 px-3 text-sm focus:outline-none focus:ring-2 focus:ring-primary/40";

const LABEL_CLS = "flex items-center gap-2 text-xs text-muted-foreground";

export function FilterBar({
  selectedGenres,
  onGenresChange,
  availableGenres,
  minRating,
  onMinRatingChange,
  certificate,
  onCertificateChange,
  availableCertificates,
  runtime,
  onRuntimeChange,
  director,
  onDirectorChange,
  availableDirectors,
  sortBy,
  onSortChange,
  onReset,
  hasActive,
}: FilterBarProps) {
  const toggleGenre = (genre: string) => {
    if (selectedGenres.includes(genre)) {
      onGenresChange(selectedGenres.filter((g) => g !== genre));
    } else {
      onGenresChange([...selectedGenres, genre]);
    }
  };

  return (
    <div className="mb-6 rounded-2xl border border-border/60 bg-card/40 backdrop-blur-sm p-4 space-y-4">
      <div className="flex items-center gap-3 flex-wrap">
        <div className="inline-flex items-center gap-2 text-sm font-medium text-muted-foreground">
          <Filter className="h-4 w-4" />
          Filters
        </div>

        <label className={LABEL_CLS}>
          Rating
          <select
            value={minRating}
            onChange={(e) => onMinRatingChange(e.target.value as MinRating)}
            className={SELECT_CLS}
          >
            <option value="all">Any</option>
            <option value="7.5">7.5+</option>
            <option value="8">8.0+</option>
            <option value="8.5">8.5+</option>
            <option value="9">9.0+</option>
          </select>
        </label>

        <label className={LABEL_CLS}>
          Certificate
          <select
            value={certificate}
            onChange={(e) => onCertificateChange(e.target.value)}
            className={SELECT_CLS}
          >
            <option value="all">All</option>
            {availableCertificates.map((c) => (
              <option key={c} value={c}>
                {c}
              </option>
            ))}
          </select>
        </label>

        <label className={LABEL_CLS}>
          Runtime
          <select
            value={runtime}
            onChange={(e) => onRuntimeChange(e.target.value as RuntimeBucket)}
            className={SELECT_CLS}
          >
            <option value="all">Any</option>
            <option value="short">Under 90 min</option>
            <option value="medium">90–120 min</option>
            <option value="long">120–150 min</option>
            <option value="epic">Over 150 min</option>
          </select>
        </label>

        <label className={LABEL_CLS}>
          Director
          <input
            type="text"
            value={director}
            onChange={(e) => onDirectorChange(e.target.value)}
            placeholder="Any director"
            list="director-options"
            className={`${SELECT_CLS} w-44`}
          />
          <datalist id="director-options">
            {availableDirectors.map((d) => (
              <option key={d} value={d} />
            ))}
          </datalist>
        </label>

        <label className={LABEL_CLS}>
          Sort
          <select
            value={sortBy}
            onChange={(e) => onSortChange(e.target.value as SortBy)}
            className={SELECT_CLS}
          >
            <option value="title">Title (A–Z)</option>
            <option value="rating-desc">Rating (high → low)</option>
            <option value="votes-desc">Most voted</option>
          </select>
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

      {availableGenres.length > 0 && (
        <div className="border-t border-border/60 pt-3">
          <div className="mb-2 text-xs text-muted-foreground">Genres</div>
          <div className="flex flex-wrap gap-2">
            {availableGenres.map((genre) => {
              const active = selectedGenres.includes(genre);
              return (
                <button
                  key={genre}
                  onClick={() => toggleGenre(genre)}
                  className={
                    "rounded-full border px-3 py-1 text-xs font-medium transition-colors " +
                    (active
                      ? "border-primary bg-primary/20 text-primary"
                      : "border-border/60 bg-secondary/40 text-muted-foreground hover:bg-secondary/70 hover:text-foreground")
                  }
                >
                  {genre}
                </button>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}
