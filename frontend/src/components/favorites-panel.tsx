import { Heart, Trash2, X } from "lucide-react";
import { useMemo, useState } from "react";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
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

    const filtered = tagFilter
      ? list.filter((x) => x.entry.tags.includes(tagFilter))
      : list;

    return filtered.sort((a, b) => {
      if (sort === "rating") return b.entry.rating - a.entry.rating;
      if (sort === "title") return a.movie.title.localeCompare(b.movie.title);
      return b.entry.addedAt - a.entry.addedAt;
    });
  }, [movies, favorites, tagFilter, sort]);

  const allUsedTags = useMemo(() => {
    const s = new Set<string>();
    Object.values(favorites).forEach((e) => e.tags.forEach((t) => s.add(t)));
    return Array.from(s).sort();
  }, [favorites]);

  const count = Object.keys(favorites).length;

  return (
    <section className="rounded-3xl border border-border/60 bg-card/60 backdrop-blur-sm shadow-[var(--shadow-card)] overflow-hidden flex flex-col">
      <div className="p-5 border-b border-border/60">
        <div className="flex items-center justify-between gap-3">
          <div className="min-w-0">
            <h2 className="text-base font-semibold flex items-center gap-2">
              <Heart className="h-4 w-4 text-rose-400 fill-rose-400" />
              Favorites
            </h2>
            <p className="text-xs text-muted-foreground mt-0.5">
              Saved for later, rated, tagged
            </p>
          </div>
          {count > 0 && <Badge variant="secondary">{count}</Badge>}
        </div>

        {count > 0 && (
          <>
            <div className="mt-3 flex items-center gap-2">
              <select
                value={sort}
                onChange={(e) => setSort(e.target.value as SortMode)}
                className="h-8 rounded-md bg-secondary/60 border border-border/60 px-2 text-xs"
              >
                <option value="recent">Recently added</option>
                <option value="rating">By rating</option>
                <option value="title">By title</option>
              </select>
              <button
                onClick={onClearAll}
                className="ml-auto inline-flex items-center gap-1 text-xs text-muted-foreground hover:text-destructive transition-colors"
              >
                <Trash2 className="h-3 w-3" />
                Clear all
              </button>
            </div>

            {allUsedTags.length > 0 && (
              <div className="mt-3 flex flex-wrap gap-1.5">
                <button
                  onClick={() => setTagFilter(null)}
                  className={cn(
                    "rounded-full px-2.5 py-0.5 text-[11px] font-medium border transition-colors",
                    tagFilter === null
                      ? "bg-primary text-primary-foreground border-primary"
                      : "border-border/60 text-muted-foreground hover:border-primary/40",
                  )}
                >
                  All
                </button>
                {allUsedTags.map((t) => (
                  <button
                    key={t}
                    onClick={() => setTagFilter(tagFilter === t ? null : t)}
                    className={cn(
                      "rounded-full px-2.5 py-0.5 text-[11px] font-medium border transition-colors",
                      tagFilter === t
                        ? "bg-primary text-primary-foreground border-primary"
                        : "border-border/60 text-muted-foreground hover:border-primary/40",
                    )}
                  >
                    {t}
                  </button>
                ))}
              </div>
            )}
          </>
        )}
      </div>

      <ScrollArea className="h-[600px]">
        <div className="p-3 space-y-2">
          {count === 0 && (
            <div className="text-center p-8 flex flex-col items-center">
              <div className="grid h-14 w-14 place-items-center rounded-2xl bg-secondary/60 mb-3">
                <Heart className="h-6 w-6 text-rose-400" />
              </div>
              <p className="text-sm font-medium">No favorites yet</p>
              <p className="mt-1 text-xs text-muted-foreground max-w-[220px]">
                Tap the ♥ on any movie to save it for later, rate it, and tag it.
              </p>
            </div>
          )}

          {entries.length === 0 && count > 0 && (
            <div className="text-center text-xs text-muted-foreground p-6">
              No favorites match this tag.
            </div>
          )}

          {entries.map(({ movie, entry }) => (
            <div
              key={movie.id}
              className="rounded-2xl border border-border/60 bg-secondary/30 hover:bg-secondary/50 p-3 transition-colors animate-in fade-in slide-in-from-right-2"
            >
              <div className="flex gap-3">
                <MoviePoster title={movie.title} year={movie.year} size="md" />
                <div className="min-w-0 flex-1">
                  <div className="flex items-start gap-2">
                    <div className="min-w-0 flex-1">
                      <h3 className="text-sm font-semibold truncate">{movie.title}</h3>
                      <p className="text-[11px] text-muted-foreground">
                        {movie.year ?? ""}
                      </p>
                    </div>
                    <button
                      onClick={() => onToggle(movie.id)}
                      className="shrink-0 text-muted-foreground hover:text-destructive transition-colors"
                      aria-label="Remove favorite"
                    >
                      <X className="h-4 w-4" />
                    </button>
                  </div>
                  <div className="mt-2 flex items-center gap-2 flex-wrap">
                    <StarRating
                      value={entry.rating}
                      onChange={(r) => onSetRating(movie.id, r)}
                    />
                    <TagPicker
                      selected={entry.tags}
                      customTags={customTags}
                      onChange={(tags) => onSetTags(movie.id, tags)}
                      onAddCustomTag={onAddCustomTag}
                    />
                  </div>
                  {entry.tags.length > 0 && (
                    <div className="mt-2 flex flex-wrap gap-1">
                      {entry.tags.map((t) => (
                        <span
                          key={t}
                          className="rounded-full bg-primary/15 text-primary text-[10px] px-2 py-0.5 font-medium"
                        >
                          {t}
                        </span>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            </div>
          ))}
        </div>
      </ScrollArea>

      {count > 0 && (
        <div className="p-4 border-t border-border/60 bg-background/30">
          <Button
            variant="secondary"
            size="sm"
            className="w-full"
            onClick={onClearAll}
          >
            <Trash2 className="h-3.5 w-3.5" />
            Clear all favorites
          </Button>
        </div>
      )}
    </section>
  );
}
