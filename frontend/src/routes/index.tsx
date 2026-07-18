import { createFileRoute } from "@tanstack/react-router";
import { useEffect, useMemo, useState } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";
import {
  Film,
  Search,
  Sparkles,
  X,
  Loader2,
  Star,
  AlertCircle,
  Heart,
  Clock,
} from "lucide-react";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import { cn } from "@/lib/utils";
import {
  fetchTitles,
  fetchRecommendations,
  type Movie,
  type Recommendation,
} from "@/lib/api/movies";
import { MoviePoster } from "@/components/movie-poster";
import {
  FilterBar,
  type SortBy,
  type RuntimeBucket,
  type MinRating,
} from "@/components/filter-bar";
import { FavoritesPanel } from "@/components/favorites-panel";
import { useFavorites } from "@/hooks/use-favorites";

export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      { title: "CineMatch — Movie Recommendations" },
      {
        name: "description",
        content:
          "Pick the movies you love, save favorites, tag them, and get AI-powered film recommendations tailored to your taste.",
      },
      { property: "og:title", content: "CineMatch — Movie Recommendations" },
      {
        property: "og:description",
        content: "AI-powered movie recommendations with favorites, tags, and ratings.",
      },
    ],
  }),
  component: Index,
});

function hasGenre(genres: string | null | undefined, selectedGenres: string[]) {
  if (!selectedGenres.length || !genres) return true;
  const genreList = genres.split(",").map((g) => g.trim());
  return selectedGenres.some((selected) =>
    genreList.some((g) => g.toLowerCase() === selected.toLowerCase())
  );
}

function inRuntime(runtime: number | null | undefined, bucket: RuntimeBucket) {
  if (bucket === "all" || runtime == null) return bucket === "all";
  if (bucket === "short") return runtime < 90;
  if (bucket === "medium") return runtime >= 90 && runtime <= 120;
  if (bucket === "long") return runtime > 120 && runtime <= 150;
  return runtime > 150; // epic
}

function Index() {
  useEffect(() => {
    document.documentElement.classList.add("dark");
  }, []);

  const [search, setSearch] = useState("");
  const [selected, setSelected] = useState<number[]>([]);
  const [selectedGenres, setSelectedGenres] = useState<string[]>([]);
  const [minRating, setMinRating] = useState<MinRating>("all");
  const [certificate, setCertificate] = useState<string>("all");
  const [runtime, setRuntime] = useState<RuntimeBucket>("all");
  const [director, setDirector] = useState<string>("");
  const [sortBy, setSortBy] = useState<SortBy>("title");

  const fav = useFavorites();

  const { data: movies = [], isLoading, isError, refetch } = useQuery({
    queryKey: ["titles"],
    queryFn: fetchTitles,
    staleTime: Infinity,
    retry: 1,
  });

  // Extract unique genres from all movies
  const availableGenres = useMemo(() => {
    const genreSet = new Set<string>();
    movies.forEach((m) => {
      if (m.genres) {
        m.genres.split(",").forEach((g) => {
          genreSet.add(g.trim());
        });
      }
    });
    return Array.from(genreSet).sort();
  }, [movies]);

  // Extract unique certificates from all movies
  const availableCertificates = useMemo(() => {
    const set = new Set<string>();
    movies.forEach((m) => {
      if (m.certificate) set.add(m.certificate);
    });
    return Array.from(set).sort();
  }, [movies]);

  // Extract unique directors from all movies
  const availableDirectors = useMemo(() => {
    const set = new Set<string>();
    movies.forEach((m) => {
      if (m.director) set.add(m.director);
    });
    return Array.from(set).sort();
  }, [movies]);

  const recMutation = useMutation({
    mutationFn: () => fetchRecommendations(selected, 8),
  });

  const filtered = useMemo(() => {
    const q = search.trim().toLowerCase();
    const minR = minRating === "all" ? 0 : parseFloat(minRating);
    const list = movies.filter(
      (m) =>
        (!q || m.title.toLowerCase().includes(q)) &&
        hasGenre(m.genres, selectedGenres) &&
        (minRating === "all" || (m.rating ?? 0) >= minR) &&
        (certificate === "all" || m.certificate === certificate) &&
        inRuntime(m.runtime, runtime) &&
        (!director.trim() ||
          (m.director ?? "").toLowerCase().includes(director.trim().toLowerCase()))
    );
    return [...list].sort((a, b) => {
      if (sortBy === "rating-desc") return (b.rating ?? 0) - (a.rating ?? 0);
      if (sortBy === "votes-desc") return (b.votes ?? 0) - (a.votes ?? 0);
      return a.title.localeCompare(b.title);
    });
  }, [movies, search, selectedGenres, minRating, certificate, runtime, director, sortBy]);

  const filteredRecs = useMemo(() => {
    if (!recMutation.data) return [];
    return recMutation.data;
  }, [recMutation.data]);

  const toggle = (id: number) =>
    setSelected((s) => (s.includes(id) ? s.filter((x) => x !== id) : [...s, id]));

  const selectedMovies = movies.filter((m) => selected.includes(m.id));
  const hasActiveFilters =
    selectedGenres.length > 0 ||
    minRating !== "all" ||
    certificate !== "all" ||
    runtime !== "all" ||
    director.trim() !== "" ||
    sortBy !== "title";

  return (
    <div className="min-h-screen bg-background text-foreground">
      <div className="pointer-events-none fixed inset-0 -z-10 overflow-hidden">
        <div className="absolute -top-40 left-1/4 h-[500px] w-[500px] rounded-full bg-primary/20 blur-[120px]" />
        <div className="absolute top-1/3 right-0 h-[400px] w-[400px] rounded-full bg-fuchsia-500/10 blur-[120px]" />
      </div>

      <header className="border-b border-border/60 backdrop-blur-xl bg-background/70 sticky top-0 z-20">
        <div className="mx-auto max-w-[1400px] px-6 py-4 flex items-center justify-between gap-4">
          <div className="flex items-center gap-3 min-w-0">
            <div className="grid h-11 w-11 shrink-0 place-items-center rounded-2xl bg-[image:var(--gradient-primary)] shadow-[var(--shadow-glow)]">
              <Film className="h-5 w-5 text-primary-foreground" />
            </div>
            <div className="min-w-0">
              <h1 className="font-heading text-xl font-extrabold tracking-tight truncate bg-gradient-to-r from-foreground to-foreground/60 bg-clip-text text-transparent">
                CineMatch
              </h1>
              <p className="text-xs text-muted-foreground truncate">
                Find your next favorite film
              </p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <Badge variant="secondary" className="hidden sm:inline-flex gap-1.5">
              <Sparkles className="h-3 w-3 text-primary" />
              {selected.length} selected
            </Badge>
            <Badge variant="secondary" className="hidden sm:inline-flex gap-1.5">
              <Heart className="h-3 w-3 text-rose-400 fill-rose-400" />
              {Object.keys(fav.favorites).length} saved
            </Badge>
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-[1400px] px-6 py-8">
        <FilterBar
          selectedGenres={selectedGenres}
          onGenresChange={setSelectedGenres}
          availableGenres={availableGenres}
          minRating={minRating}
          onMinRatingChange={setMinRating}
          certificate={certificate}
          onCertificateChange={setCertificate}
          availableCertificates={availableCertificates}
          runtime={runtime}
          onRuntimeChange={setRuntime}
          director={director}
          onDirectorChange={setDirector}
          availableDirectors={availableDirectors}
          sortBy={sortBy}
          onSortChange={setSortBy}
          hasActive={hasActiveFilters}
          onReset={() => {
            setSelectedGenres([]);
            setMinRating("all");
            setCertificate("all");
            setRuntime("all");
            setDirector("");
            setSortBy("title");
          }}
        />

        <div className="grid gap-6 lg:grid-cols-2 xl:grid-cols-3">
          {/* Library */}
          <section className="rounded-3xl border border-border/60 bg-card/60 backdrop-blur-sm shadow-[var(--shadow-card)] overflow-hidden flex flex-col">
            <div className="p-5 border-b border-border/60 space-y-4">
              <div className="flex items-center justify-between gap-2">
                <h2 className="text-base font-semibold flex items-center gap-2">
                  <Sparkles className="h-4 w-4 text-primary" />
                  Your library
                </h2>
                <span className="text-xs text-muted-foreground">
                  {selected.length} selected
                </span>
              </div>
              <div className="relative">
                <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                <Input
                  value={search}
                  onChange={(e) => setSearch(e.target.value)}
                  placeholder="Search movies..."
                  className="pl-9 bg-secondary/50 border-border/60 h-11"
                />
              </div>

              {selectedMovies.length > 0 && (
                <div className="flex flex-wrap gap-2">
                  {selectedMovies.slice(0, 4).map((m) => (
                    <button
                      key={m.id}
                      onClick={() => toggle(m.id)}
                      className="group inline-flex items-center gap-1.5 rounded-full bg-primary/15 hover:bg-primary/25 px-3 py-1 text-xs font-medium text-primary transition-colors"
                    >
                      <span className="max-w-[120px] truncate">{m.title}</span>
                      <X className="h-3 w-3 opacity-60 group-hover:opacity-100" />
                    </button>
                  ))}
                  {selectedMovies.length > 4 && (
                    <span className="text-xs text-muted-foreground self-center">
                      +{selectedMovies.length - 4}
                    </span>
                  )}
                  <button
                    onClick={() => setSelected([])}
                    className="ml-auto text-xs text-muted-foreground hover:text-foreground transition-colors"
                  >
                    Clear
                  </button>
                </div>
              )}
            </div>

            <ScrollArea className="h-[520px]">
              <div className="p-3 space-y-2">
                {isLoading &&
                  Array.from({ length: 6 }).map((_, i) => (
                    <div
                      key={i}
                      className="h-20 rounded-2xl bg-secondary/40 animate-pulse"
                    />
                  ))}

                {isError && (
                  <div className="flex flex-col items-center text-center gap-3 p-6 rounded-2xl border border-destructive/30 bg-destructive/10">
                    <AlertCircle className="h-6 w-6 text-destructive" />
                    <p className="text-sm font-medium">Couldn't load movies</p>
                    <Button size="sm" variant="secondary" onClick={() => refetch()}>
                      Retry
                    </Button>
                  </div>
                )}

                {!isLoading &&
                  !isError &&
                  filtered.map((m) => (
                    <MovieCard
                      key={m.id}
                      movie={m}
                      selected={selected.includes(m.id)}
                      favorited={fav.isFavorite(m.id)}
                      onClick={() => toggle(m.id)}
                      onToggleFav={() => fav.toggle(m.id)}
                    />
                  ))}

                {!isLoading && !isError && filtered.length === 0 && (
                  <div className="text-center text-sm text-muted-foreground p-8">
                    No movies match your filters
                  </div>
                )}
              </div>
            </ScrollArea>

            <div className="p-5 border-t border-border/60 bg-background/30">
              <Button
                size="lg"
                disabled={selected.length === 0 || recMutation.isPending}
                onClick={() => recMutation.mutate()}
                className="w-full h-12 bg-[image:var(--gradient-primary)] hover:opacity-90 text-primary-foreground font-semibold shadow-[var(--shadow-glow)] transition-all"
              >
                {recMutation.isPending ? (
                  <>
                    <Loader2 className="h-4 w-4 animate-spin" />
                    Finding matches...
                  </>
                ) : (
                  <>
                    <Sparkles className="h-4 w-4" />
                    Get Recommendations
                  </>
                )}
              </Button>
            </div>
          </section>

          {/* Recommendations */}
          <section className="rounded-3xl border border-border/60 bg-card/60 backdrop-blur-sm shadow-[var(--shadow-card)] overflow-hidden flex flex-col">
            <div className="p-5 border-b border-border/60 flex items-center justify-between gap-3">
              <div className="min-w-0">
                <h2 className="text-base font-semibold flex items-center gap-2">
                  <Star className="h-4 w-4 text-primary" />
                  Recommended for you
                </h2>
                <p className="text-xs text-muted-foreground mt-0.5">
                  Ranked by similarity to your picks
                </p>
              </div>
              {recMutation.data && (
                <Badge variant="secondary">
                  {filteredRecs.length}
                </Badge>
              )}
            </div>

            <div className="flex-1 p-4">
              {recMutation.isIdle && <EmptyState />}

              {recMutation.isPending && (
                <div className="space-y-3">
                  {Array.from({ length: 4 }).map((_, i) => (
                    <div
                      key={i}
                      className="h-24 rounded-2xl bg-secondary/40 animate-pulse"
                    />
                  ))}
                </div>
              )}

              {recMutation.isError && (
                <div className="flex flex-col items-center text-center gap-3 p-8 rounded-2xl border border-destructive/30 bg-destructive/10">
                  <AlertCircle className="h-6 w-6 text-destructive" />
                  <p className="text-sm">
                    {(recMutation.error as Error)?.message ?? "Something went wrong"}
                  </p>
                  <Button
                    size="sm"
                    variant="secondary"
                    onClick={() => recMutation.mutate()}
                  >
                    Try again
                  </Button>
                </div>
              )}

              {recMutation.data && filteredRecs.length > 0 && (
                <div className="space-y-3">
                  {filteredRecs.map((r, i) => (
                    <RecCard
                      key={r.id}
                      rec={r}
                      rank={i + 1}
                      favorited={fav.isFavorite(r.id)}
                      onToggleFav={() => fav.toggle(r.id)}
                    />
                  ))}
                </div>
              )}

              {recMutation.data && filteredRecs.length === 0 && (
                <div className="text-center text-sm text-muted-foreground p-8">
                  {recMutation.data.length === 0
                    ? "No recommendations found. Try selecting different movies."
                    : "No recommendations available."}
                </div>
              )}
            </div>
          </section>

          {/* Favorites */}
          <div className="lg:col-span-2 xl:col-span-1">
            <FavoritesPanel
              movies={movies}
              favorites={fav.favorites}
              customTags={fav.customTags}
              onToggle={fav.toggle}
              onSetRating={fav.setRating}
              onSetTags={fav.setTags}
              onAddCustomTag={fav.addCustomTag}
              onClearAll={fav.clearAll}
            />
          </div>
        </div>
      </main>
    </div>
  );
}

function FavButton({
  active,
  onClick,
}: {
  active: boolean;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      onClick={(e) => {
        e.stopPropagation();
        onClick();
      }}
      className={cn(
        "shrink-0 grid h-8 w-8 place-items-center rounded-full transition-all",
        active
          ? "bg-rose-500/15 text-rose-400 hover:bg-rose-500/25"
          : "bg-background/40 text-muted-foreground hover:text-rose-400 hover:bg-rose-500/10",
      )}
      aria-label={active ? "Remove from favorites" : "Add to favorites"}
    >
      <Heart className={cn("h-4 w-4", active && "fill-rose-400")} />
    </button>
  );
}

function MovieCard({
  movie,
  selected,
  favorited,
  onClick,
  onToggleFav,
}: {
  movie: Movie;
  selected: boolean;
  favorited: boolean;
  onClick: () => void;
  onToggleFav: () => void;
}) {
  return (
    <button
      onClick={onClick}
      className={cn(
        "group w-full text-left rounded-2xl border p-3 transition-all duration-200",
        "hover:-translate-y-0.5 hover:shadow-[var(--shadow-card)]",
        selected
          ? "border-primary/70 bg-primary/10 shadow-[var(--shadow-glow)]"
          : "border-border/60 bg-secondary/30 hover:bg-secondary/60 hover:border-primary/40",
      )}
    >
      <div className="flex items-start gap-3">
        <MoviePoster title={movie.title} year={movie.year} posterUrl={movie.poster_url} size="sm" />
        <div className="min-w-0 flex-1">
          <h3 className="font-semibold leading-snug break-words line-clamp-1">
            {movie.title}
          </h3>
          <div className="mt-1 flex items-center gap-2.5 text-[11px] text-muted-foreground">
            {movie.rating != null && (
              <span className="inline-flex items-center gap-1 font-medium text-amber-400">
                <Star className="h-3 w-3 fill-amber-400" />
                {movie.rating.toFixed(1)}
              </span>
            )}
            {movie.runtime != null && (
              <span className="inline-flex items-center gap-1">
                <Clock className="h-3 w-3" />
                {movie.runtime}m
              </span>
            )}
            {movie.year && <span>{movie.year}</span>}
          </div>
          {movie.short && (
            <p className="mt-1.5 text-xs text-muted-foreground/90 line-clamp-2 break-words">
              {movie.short}
            </p>
          )}
        </div>
        <div className="shrink-0 flex flex-col items-center gap-1.5">
          <FavButton active={favorited} onClick={onToggleFav} />
          <div
            className={cn(
              "grid h-5 w-5 place-items-center rounded-full border transition-all",
              selected
                ? "bg-primary border-primary text-primary-foreground scale-110"
                : "border-border bg-background/40 group-hover:border-primary/50",
            )}
          >
            {selected && <span className="text-[10px] leading-none">✓</span>}
          </div>
        </div>
      </div>
    </button>
  );
}

function RecCard({
  rec,
  rank,
  favorited,
  onToggleFav,
}: {
  rec: Recommendation;
  rank: number;
  favorited: boolean;
  onToggleFav: () => void;
}) {
  const pct = Math.round(rec.score * 100);
  return (
    <div
      className="rounded-2xl border border-border/60 bg-secondary/40 p-3 transition-all hover:border-primary/40 hover:bg-secondary/60 animate-in fade-in slide-in-from-bottom-2"
      style={{ animationDelay: `${rank * 50}ms`, animationFillMode: "backwards" }}
    >
      <div className="flex items-start gap-3">
        <div className="relative">
          <MoviePoster title={rec.title} year={rec.year} posterUrl={rec.poster_url} size="md" />
          <div className="absolute -top-1.5 -left-1.5 grid h-6 w-6 place-items-center rounded-full bg-[image:var(--gradient-primary)] text-[11px] font-bold text-primary-foreground shadow-[var(--shadow-glow)]">
            {rank}
          </div>
        </div>
        <div className="min-w-0 flex-1">
          <div className="flex items-start justify-between gap-2">
            <div className="min-w-0">
              <h3 className="font-semibold truncate">{rec.title}</h3>
              {rec.year && (
                <p className="text-[11px] text-muted-foreground">{rec.year}</p>
              )}
            </div>
            <div className="flex items-center gap-1.5 shrink-0">
              <div className="text-right">
                <div className="text-base font-extrabold text-primary tabular-nums leading-none">
                  {pct}
                  <span className="text-[10px] font-semibold">%</span>
                </div>
                <div className="text-[9px] uppercase tracking-wider text-muted-foreground">
                  match
                </div>
              </div>
              <FavButton active={favorited} onClick={onToggleFav} />
            </div>
          </div>
          {rec.short && (
            <p className="mt-1 text-xs text-muted-foreground line-clamp-2">
              {rec.short}
            </p>
          )}
          <div className="mt-2 h-1.5 w-full rounded-full bg-background/60 overflow-hidden">
            <div
              className="h-full rounded-full bg-[image:var(--gradient-primary)] transition-all"
              style={{ width: `${pct}%` }}
            />
          </div>
        </div>
      </div>
    </div>
  );
}

function EmptyState() {
  return (
    <div className="h-full min-h-[400px] flex flex-col items-center justify-center text-center p-8">
      <div className="grid h-16 w-16 place-items-center rounded-2xl bg-secondary/60 mb-4">
        <Sparkles className="h-7 w-7 text-primary" />
      </div>
      <h3 className="font-semibold">Select movies to get recommendations</h3>
      <p className="mt-2 text-sm text-muted-foreground max-w-xs">
        Pick a few films you love from your library and we'll find your next favorite.
      </p>
    </div>
  );
}
