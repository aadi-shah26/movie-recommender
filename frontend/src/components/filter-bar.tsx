import { cn } from "@/lib/utils";

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

const FIELD_CLS =
  "h-8 w-full rounded-md border border-input bg-background px-2 text-xs text-foreground focus:outline-none focus:ring-1 focus:ring-ring";

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <label className="flex flex-col gap-1 text-[11px] text-muted-foreground">
      {label}
      {children}
    </label>
  );
}

/** Compact filter controls; the parent decides when to show them. */
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
  const toggleGenre = (genre: string) =>
    onGenresChange(
      selectedGenres.includes(genre)
        ? selectedGenres.filter((g) => g !== genre)
        : [...selectedGenres, genre],
    );

  return (
    <div className="space-y-4">
      {availableGenres.length > 0 && (
        <div className="flex flex-wrap gap-1">
          {availableGenres.map((genre) => {
            const active = selectedGenres.includes(genre);
            return (
              <button
                key={genre}
                type="button"
                onClick={() => toggleGenre(genre)}
                aria-pressed={active}
                className={cn(
                  "rounded-md border px-2 py-0.5 text-[11px] transition-colors",
                  active
                    ? "border-primary/60 bg-primary/10 text-primary"
                    : "border-border text-muted-foreground hover:text-foreground hover:border-foreground/20",
                )}
              >
                {genre}
              </button>
            );
          })}
        </div>
      )}

      <div className="grid grid-cols-2 gap-x-3 gap-y-2.5">
        <Field label="Sort by">
          <select
            value={sortBy}
            onChange={(e) => onSortChange(e.target.value as SortBy)}
            className={FIELD_CLS}
          >
            <option value="title">Title</option>
            <option value="rating-desc">IMDb rating</option>
            <option value="votes-desc">Most voted</option>
          </select>
        </Field>
        <Field label="Minimum rating">
          <select
            value={minRating}
            onChange={(e) => onMinRatingChange(e.target.value as MinRating)}
            className={FIELD_CLS}
          >
            <option value="all">Any</option>
            <option value="7.5">7.5+</option>
            <option value="8">8.0+</option>
            <option value="8.5">8.5+</option>
            <option value="9">9.0+</option>
          </select>
        </Field>
        <Field label="Runtime">
          <select
            value={runtime}
            onChange={(e) => onRuntimeChange(e.target.value as RuntimeBucket)}
            className={FIELD_CLS}
          >
            <option value="all">Any</option>
            <option value="short">Under 90 min</option>
            <option value="medium">90–120 min</option>
            <option value="long">120–150 min</option>
            <option value="epic">Over 150 min</option>
          </select>
        </Field>
        <Field label="Certificate">
          <select
            value={certificate}
            onChange={(e) => onCertificateChange(e.target.value)}
            className={FIELD_CLS}
          >
            <option value="all">Any</option>
            {availableCertificates.map((c) => (
              <option key={c} value={c}>
                {c}
              </option>
            ))}
          </select>
        </Field>
        <div className="col-span-2">
          <Field label="Director">
            <input
              type="text"
              value={director}
              onChange={(e) => onDirectorChange(e.target.value)}
              placeholder="Any director"
              list="director-options"
              className={FIELD_CLS}
            />
            <datalist id="director-options">
              {availableDirectors.map((d) => (
                <option key={d} value={d} />
              ))}
            </datalist>
          </Field>
        </div>
      </div>

      {hasActive && (
        <button
          type="button"
          onClick={onReset}
          className="text-[11px] text-muted-foreground underline underline-offset-2 hover:text-foreground"
        >
          Reset filters
        </button>
      )}
    </div>
  );
}
