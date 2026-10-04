import { X } from "lucide-react";
import { useMemo, useState } from "react";
import { cn } from "@/lib/utils";
import { MoviePoster } from "./movie-poster";
import { StarRating } from "./star-rating";
import { TagPicker } from "./tag-picker";
import type { Movie } from "@/lib/api/movies";
import type { FavoritesMap } from "@/lib/storage/favorites";

interface FavoritesPanelProps {
  movies: Movie[];
  favorites: FavoritesMap;
  customTags: string[];
  onToggle: (id: number) => void;
  onSetRating: (id: number, r: number) => void;
  onSetTags: (id: number, tags: string[]) => void;
  onAddCustomTag: (t: string) => void;
  onClearAll: () => void;
}

type SortMode = "recent" | "rating" | "title";

const SORTS: { value: SortMode; label: string }[] = [
  { value: "recent", label: "Recent" },
  { value: "rating", label: "Your rating" },
  { value: "title", label: "Title" },
];

/** The "Saved" tab: favorites with your own rating and tags. */
export function FavoritesPanel({
  movies,
  favorites,
  customTags,
  onToggle,
  onSetRating,
  onSetTags,
  onAddCustomTag,
  onClearAll,
}: FavoritesPanelProps) {
  const [tagFilter, setTagFilter] = useState<string | null>(null);
  const [sort, setSort] = useState<SortMode>("recent");

  const entries = useMemo(() => {
    const list = movies
      .filter((m) => favorites[m.id])
      .map((m) => ({ movie: m, entry: favorites[m.id]! }));
    const filtered = tagFilter ? list.filter((x) => x.entry.tags.includes(tagFilter)) : list;
    return filtered.sort((a, b) => {
      if (sort === "rating") return b.entry.rating - a.entry.rating;
      if (sort === "title") return a.movie.title.localeCompare(b.movie.title);
      return b.entry.addedAt - a.entry.addedAt;
    });
  }, [movies, favorites, tagFilter, sort]);

  const usedTags = useMemo(() => {
    const s = new Set<string>();
    Object.values(favorites).forEach((e) => e.tags.forEach((t) => s.add(t)));
    return Array.from(s).sort();
  }, [favorites]);

  const count = Object.keys(favorites).length;

  if (count === 0) {
    return (
      <div className="py-24 text-center">
        <p className="font-heading text-3xl text-foreground/90">Nothing saved yet</p>
        <p className="mt-2 text-sm text-muted-foreground">
          Use the heart on any film to keep it here, then rate and tag it.
        </p>
      </div>
    );
  }

  return (
    <div>
      <div className="flex flex-wrap items-center gap-x-4 gap-y-2 pb-4 border-b border-border">
        <div className="flex items-center gap-3 text-xs">
          {SORTS.map((s) => (
            <button
              key={s.value}
              type="button"
              onClick={() => setSort(s.value)}
              className={cn(
                "transition-colors",
                sort === s.value
                  ? "text-foreground"
                  : "text-muted-foreground hover:text-foreground",
              )}
            >
              {s.label}
            </button>
          ))}
        </div>
        {usedTags.length > 0 && (
          <div className="flex flex-wrap gap-1">
            {usedTags.map((t) => (
              <button
                key={t}
                type="button"
                onClick={() => setTagFilter(tagFilter === t ? null : t)}
                aria-pressed={tagFilter === t}
                className={cn(
                  "rounded-md border px-2 py-0.5 text-[11px] transition-colors",
                  tagFilter === t
                    ? "border-primary/60 bg-primary/10 text-primary"
                    : "border-border text-muted-foreground hover:text-foreground",
                )}
              >
                {t}
              </button>
            ))}
          </div>
        )}
        <button
          type="button"
          onClick={onClearAll}
          className="ml-auto text-xs text-muted-foreground hover:text-destructive transition-colors"
        >
          Clear all
        </button>
      </div>

      {entries.length === 0 && (
        <p className="py-12 text-center text-sm text-muted-foreground">
          No saved films with this tag.
        </p>
      )}

      <ul className="divide-y divide-border">
        {entries.map(({ movie, entry }) => (
          <li key={movie.id} className="flex gap-4 py-4">
            <MoviePoster
              title={movie.title}
              year={movie.year}
              posterUrl={movie.poster_url}
              size="md"
            />
            <div className="min-w-0 flex-1">
              <div className="flex items-start gap-3">
                <div className="min-w-0 flex-1">
                  <h3 className="text-sm font-medium truncate">{movie.title}</h3>
                  <p className="mt-0.5 text-xs text-muted-foreground">
                    {[movie.year, movie.director].filter(Boolean).join(" · ")}
                  </p>
                </div>
                <button
                  type="button"
                  onClick={() => onToggle(movie.id)}
                  className="shrink-0 text-muted-foreground hover:text-foreground transition-colors"
                  aria-label={`Remove ${movie.title} from saved`}
                >
                  <X className="h-4 w-4" />
                </button>
              </div>
              <div className="mt-2.5 flex flex-wrap items-center gap-3">
                <StarRating value={entry.rating} onChange={(r) => onSetRating(movie.id, r)} />
                <TagPicker
                  selected={entry.tags}
                  customTags={customTags}
                  onChange={(tags) => onSetTags(movie.id, tags)}
                  onAddCustomTag={onAddCustomTag}
                />
                {entry.tags.map((t) => (
                  <span key={t} className="text-[11px] text-muted-foreground">
                    #{t}
                  </span>
                ))}
              </div>
            </div>
          </li>
        ))}
      </ul>
    </div>
  );
}
