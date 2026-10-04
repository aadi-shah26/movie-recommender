import { createFileRoute } from "@tanstack/react-router";
import { useMemo, useState } from "react";
import { useMutation, useQuery, type UseMutationResult } from "@tanstack/react-query";
import { Check, ChevronDown, Heart, Loader2, Search, X } from "lucide-react";
import { Button } from "@/components/ui/button";
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
      { title: "Movie Recommender" },
      {
        name: "description",
        content: "Pick a few films you love and get personalized movie recommendations.",
      },
      { property: "og:title", content: "Movie Recommender" },
    ],
  }),
  component: Index,
});

const N_RECOMMENDATIONS = 12;
const QUICK_PICKS = [
  "The Godfather",
  "Spirited Away",
  "Pulp Fiction",
  "Amélie",
  "Parasite",
  "In the Mood for Love",
];

function hasGenre(genres: string | null | undefined, selectedGenres: string[]) {
  if (!selectedGenres.length || !genres) return true;
  const genreList = genres.split(",").map((g) => g.trim().toLowerCase());
  return selectedGenres.some((s) => genreList.includes(s.toLowerCase()));
}

function inRuntime(runtime: number | null | undefined, bucket: RuntimeBucket) {
  if (bucket === "all") return true;
  if (runtime == null) return false;
  if (bucket === "short") return runtime < 90;
  if (bucket === "medium") return runtime >= 90 && runtime <= 120;
  if (bucket === "long") return runtime > 120 && runtime <= 150;
  return runtime > 150;
}

function sameIds(a: number[], b: number[]) {
  return a.length === b.length && [...a].sort().every((x, i) => x === [...b].sort()[i]);
}

function Index() {
  const [search, setSearch] = useState("");
  const [selected, setSelected] = useState<number[]>([]);
  const [showFilters, setShowFilters] = useState(false);
  const [selectedGenres, setSelectedGenres] = useState<string[]>([]);
  const [minRating, setMinRating] = useState<MinRating>("all");
  const [certificate, setCertificate] = useState<string>("all");
  const [runtime, setRuntime] = useState<RuntimeBucket>("all");
  const [director, setDirector] = useState<string>("");
  const [sortBy, setSortBy] = useState<SortBy>("title");
  const [tab, setTab] = useState<"recs" | "saved">("recs");

  const fav = useFavorites();

  const {
    data: movies = [],
    isLoading,
    isError,
    refetch,
  } = useQuery({ queryKey: ["titles"], queryFn: fetchTitles, staleTime: Infinity, retry: 1 });

  const byId = useMemo(() => new Map(movies.map((m) => [m.id, m])), [movies]);

  const options = useMemo(() => {
    const genres = new Set<string>();
    const certificates = new Set<string>();
    const directors = new Set<string>();
    movies.forEach((m) => {
      m.genres?.split(",").forEach((g) => genres.add(g.trim()));
      if (m.certificate) certificates.add(m.certificate);
      if (m.director) directors.add(m.director);
    });
    return {
      genres: [...genres].sort(),
      certificates: [...certificates].sort(),
      directors: [...directors].sort(),
    };
  }, [movies]);

  const quickPicks = useMemo(
    () => QUICK_PICKS.map((t) => movies.find((m) => m.title === t)).filter((m): m is Movie => !!m),
    [movies],
  );

  const recMutation = useMutation({
    mutationFn: (ids: number[]) => fetchRecommendations(ids, N_RECOMMENDATIONS),
  });

  const filtered = useMemo(() => {
    const q = search.trim().toLowerCase();
    const minR = minRating === "all" ? 0 : parseFloat(minRating);
    const dir = director.trim().toLowerCase();
    const list = movies.filter(
      (m) =>
        (!q || m.title.toLowerCase().includes(q)) &&
        hasGenre(m.genres, selectedGenres) &&
        (m.rating ?? 0) >= minR &&
        (certificate === "all" || m.certificate === certificate) &&
        inRuntime(m.runtime, runtime) &&
        (!dir || (m.director ?? "").toLowerCase().includes(dir)),
    );
    return list.sort((a, b) => {
      if (sortBy === "rating-desc") return (b.rating ?? 0) - (a.rating ?? 0);
      if (sortBy === "votes-desc") return (b.votes ?? 0) - (a.votes ?? 0);
      return a.title.localeCompare(b.title);
    });
  }, [movies, search, selectedGenres, minRating, certificate, runtime, director, sortBy]);

  const activeFilters =
    selectedGenres.length +
    Number(minRating !== "all") +
    Number(certificate !== "all") +
    Number(runtime !== "all") +
    Number(director.trim() !== "");

  const toggle = (id: number) =>
    setSelected((s) => (s.includes(id) ? s.filter((x) => x !== id) : [...s, id]));

  const recommend = () => {
    setTab("recs");
    recMutation.mutate(selected);
  };

  const selectedMovies = selected.map((id) => byId.get(id)).filter((m): m is Movie => !!m);
  const stale = !!recMutation.variables && !sameIds(recMutation.variables, selected);
  const savedCount = Object.keys(fav.favorites).length;

  return (
    <div className="min-h-screen bg-background text-foreground">
      <header className="border-b border-border">
        <div className="mx-auto flex h-16 max-w-[1320px] items-center justify-between gap-4 px-4 sm:px-6">
          <span className="font-heading text-[28px] leading-none tracking-tight">
            Movie Recommender
          </span>
        </div>
      </header>

      <main className="mx-auto grid max-w-[1320px] grid-cols-[minmax(0,1fr)] gap-10 px-4 py-8 sm:px-6 lg:grid-cols-[340px_minmax(0,1fr)] lg:gap-12">
        {/* Library */}
        <aside className="flex flex-col lg:sticky lg:top-8 lg:h-[calc(100vh-8rem)]">
          <div className="flex items-baseline justify-between">
            <h2 className="font-heading text-2xl">Library</h2>
            <span className="text-xs text-muted-foreground tabular-nums">
              {isLoading ? "" : `${filtered.length} of ${movies.length}`}
            </span>
          </div>

          <div className="relative mt-4">
            <Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
            <input
              type="search"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="Search titles"
              className="h-10 w-full rounded-md border border-input bg-card pl-9 pr-3 text-sm placeholder:text-muted-foreground focus:outline-none focus:ring-1 focus:ring-ring"
            />
          </div>

          <button
            type="button"
            onClick={() => setShowFilters((v) => !v)}
            aria-expanded={showFilters}
            className="mt-3 flex items-center gap-1.5 self-start text-xs text-muted-foreground hover:text-foreground"
          >
            Filters
            {activeFilters > 0 && <span className="text-primary">({activeFilters})</span>}
            <ChevronDown
              className={cn("h-3.5 w-3.5 transition-transform", showFilters && "rotate-180")}
            />
          </button>
          {showFilters && (
            <div className="mt-3 border-b border-border pb-4">
              <FilterBar
                selectedGenres={selectedGenres}
                onGenresChange={setSelectedGenres}
                availableGenres={options.genres}
                minRating={minRating}
                onMinRatingChange={setMinRating}
                certificate={certificate}
                onCertificateChange={setCertificate}
                availableCertificates={options.certificates}
                runtime={runtime}
                onRuntimeChange={setRuntime}
                director={director}
                onDirectorChange={setDirector}
                availableDirectors={options.directors}
                sortBy={sortBy}
                onSortChange={setSortBy}
                hasActive={activeFilters > 0 || sortBy !== "title"}
                onReset={() => {
                  setSelectedGenres([]);
                  setMinRating("all");
                  setCertificate("all");
                  setRuntime("all");
                  setDirector("");
                  setSortBy("title");
                }}
              />
            </div>
          )}

          <ul className="mt-3 -mx-2 h-[480px] overflow-y-auto lg:h-auto lg:flex-1">
            {isLoading &&
              Array.from({ length: 8 }).map((_, i) => (
                <li key={i} className="flex items-center gap-3 px-2 py-2">
                  <div className="h-[3.75rem] w-10 animate-pulse rounded-[3px] bg-secondary" />
                  <div className="h-3 w-2/3 animate-pulse rounded bg-secondary" />
                </li>
              ))}
            {isError && (
              <li className="px-2 py-8 text-center text-sm text-muted-foreground">
                Couldn't load the library.{" "}
                <button className="underline underline-offset-2" onClick={() => refetch()}>
                  Retry
                </button>
              </li>
            )}
            {!isLoading && !isError && filtered.length === 0 && (
              <li className="px-2 py-8 text-center text-sm text-muted-foreground">
                No films match these filters.
              </li>
            )}
            {filtered.map((m) => (
              <LibraryRow
                key={m.id}
                movie={m}
                selected={selected.includes(m.id)}
                saved={fav.isFavorite(m.id)}
                onSelect={() => toggle(m.id)}
                onSave={() => fav.toggle(m.id)}
              />
            ))}
          </ul>
        </aside>

        {/* Picks + results */}
        <section className="min-w-0">
          <h1 className="font-heading text-4xl leading-tight sm:text-5xl">
            Pick a few films you love.
          </h1>
          <p className="mt-2 max-w-xl text-sm text-muted-foreground">
            We'll rank the rest of the catalog by what viewers with similar taste went on to love.
          </p>

          <div className="mt-8 rounded-lg border border-border bg-card p-4">
            <div className="flex flex-wrap items-center gap-2">
              {selectedMovies.length === 0 ? (
                <>
                  <span className="mr-1 text-xs text-muted-foreground">Start with</span>
                  {quickPicks.map((m) => (
                    <button
                      key={m.id}
                      type="button"
                      onClick={() => toggle(m.id)}
                      className="rounded-md border border-border px-2.5 py-1 text-xs text-muted-foreground transition-colors hover:border-foreground/25 hover:text-foreground"
                    >
                      {m.title}
                    </button>
                  ))}
                </>
              ) : (
                selectedMovies.map((m) => (
                  <span
                    key={m.id}
                    className="inline-flex items-center gap-2 rounded-md border border-border bg-background py-1 pl-1 pr-1.5 text-xs"
                  >
                    <MoviePoster
                      title={m.title}
                      posterUrl={m.poster_url}
                      size="sm"
                      className="h-7 w-5"
                    />
                    <span className="max-w-[160px] truncate">{m.title}</span>
                    <button
                      type="button"
                      onClick={() => toggle(m.id)}
                      className="text-muted-foreground hover:text-foreground"
                      aria-label={`Remove ${m.title}`}
                    >
                      <X className="h-3.5 w-3.5" />
                    </button>
                  </span>
                ))
              )}
            </div>
            <div className="mt-4 flex items-center gap-4">
              <Button
                data-action="recommend"
                disabled={selected.length === 0 || recMutation.isPending}
                onClick={recommend}
                className="h-10 px-5 disabled:bg-secondary disabled:text-muted-foreground disabled:opacity-100 disabled:shadow-none"
              >
                {recMutation.isPending && <Loader2 className="h-4 w-4 animate-spin" />}
                {stale ? "Update recommendations" : "Recommend"}
              </Button>
              {selected.length > 0 && (
                <button
                  type="button"
                  onClick={() => setSelected([])}
                  className="text-xs text-muted-foreground hover:text-foreground"
                >
                  Clear picks
                </button>
              )}
              <span className="ml-auto text-xs text-muted-foreground tabular-nums">
                {selected.length} {selected.length === 1 ? "pick" : "picks"}
              </span>
            </div>
          </div>

          <div className="mt-10 flex gap-6 border-b border-border" role="tablist">
            <TabButton active={tab === "recs"} onClick={() => setTab("recs")}>
              Recommended
            </TabButton>
            <TabButton active={tab === "saved"} onClick={() => setTab("saved")}>
              Saved{savedCount > 0 && <span className="ml-1.5 tabular-nums">{savedCount}</span>}
            </TabButton>
          </div>

          <div className="pt-6">
            {tab === "saved" ? (
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
            ) : (
              <Recommendations
                state={recMutation}
                isSaved={fav.isFavorite}
                onSave={fav.toggle}
                onRetry={recommend}
              />
            )}
          </div>
        </section>
      </main>
    </div>
  );
}

function TabButton({
  active,
  onClick,
  children,
}: {
  active: boolean;
  onClick: () => void;
  children: React.ReactNode;
}) {
  return (
    <button
      type="button"
      role="tab"
      aria-selected={active}
      onClick={onClick}
      className={cn(
        "-mb-px border-b pb-3 text-sm transition-colors",
        active
          ? "border-foreground text-foreground"
          : "border-transparent text-muted-foreground hover:text-foreground",
      )}
    >
      {children}
    </button>
  );
}

function SaveButton({
  saved,
  onClick,
  title,
  className,
}: {
  saved: boolean;
  onClick: () => void;
  title: string;
  className?: string;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      aria-pressed={saved}
      aria-label={saved ? `Remove ${title} from saved` : `Save ${title}`}
      className={cn(
        "grid h-8 w-8 shrink-0 place-items-center rounded-md transition-colors",
        saved ? "text-primary" : "text-muted-foreground hover:text-foreground",
        className,
      )}
    >
      <Heart className={cn("h-4 w-4", saved && "fill-primary")} />
    </button>
  );
}

function LibraryRow({
  movie,
  selected,
  saved,
  onSelect,
  onSave,
}: {
  movie: Movie;
  selected: boolean;
  saved: boolean;
  onSelect: () => void;
  onSave: () => void;
}) {
  const meta = [movie.year, movie.runtime && `${movie.runtime} min`].filter(Boolean).join(" · ");
  return (
    <li
      className={cn(
        "group flex items-center rounded-md pr-1 transition-colors",
        selected ? "bg-primary/[0.08]" : "hover:bg-accent/60",
      )}
    >
      <button
        type="button"
        data-movie-id={movie.id}
        onClick={onSelect}
        aria-pressed={selected}
        className="flex min-w-0 flex-1 items-center gap-3 px-2 py-2 text-left"
      >
        <MoviePoster title={movie.title} year={movie.year} posterUrl={movie.poster_url} size="sm" />
        <span className="min-w-0 flex-1">
          <span className="block truncate text-sm">{movie.title}</span>
          <span className="mt-0.5 block text-xs text-muted-foreground tabular-nums">
            {meta}
            {movie.rating != null && (
              <span className="ml-2 text-foreground/70">★ {movie.rating.toFixed(1)}</span>
            )}
          </span>
        </span>
        <span
          className={cn(
            "grid h-5 w-5 shrink-0 place-items-center rounded-full border transition-colors",
            selected
              ? "border-primary bg-primary text-primary-foreground"
              : "border-border group-hover:border-foreground/30",
          )}
        >
          {selected && <Check className="h-3 w-3" strokeWidth={3} />}
        </span>
      </button>
      <SaveButton
        saved={saved}
        onClick={onSave}
        title={movie.title}
        className={cn(!saved && "opacity-0 group-hover:opacity-100 focus-visible:opacity-100")}
      />
    </li>
  );
}

function Recommendations({
  state,
  isSaved,
  onSave,
  onRetry,
}: {
  state: UseMutationResult<Recommendation[], Error, number[]>;
  isSaved: (id: number) => boolean;
  onSave: (id: number) => void;
  onRetry: () => void;
}) {
  if (state.isIdle) {
    return (
      <div className="py-24 text-center">
        <p className="font-heading text-3xl text-foreground/90">
          Your recommendations will appear here
        </p>
        <p className="mt-2 text-sm text-muted-foreground">
          Choose one or more films from the library, then select Recommend.
        </p>
      </div>
    );
  }

  if (state.isError) {
    return (
      <div className="py-24 text-center">
        <p className="text-sm text-muted-foreground">
          Couldn't get recommendations: {state.error?.message ?? "unknown error"}
        </p>
        <button className="mt-3 text-sm underline underline-offset-2" onClick={onRetry}>
          Try again
        </button>
      </div>
    );
  }

  return (
    <div className="grid grid-cols-2 gap-x-5 gap-y-8 sm:grid-cols-3 xl:grid-cols-4">
      {state.isPending &&
        Array.from({ length: N_RECOMMENDATIONS }).map((_, i) => (
          <div key={i}>
            <div className="aspect-[2/3] animate-pulse rounded bg-secondary" />
            <div className="mt-3 h-3 w-3/4 animate-pulse rounded bg-secondary" />
          </div>
        ))}
      {!state.isPending &&
        state.data?.map((rec, i) => (
          <RecCard
            key={rec.id}
            rec={rec}
            rank={i + 1}
            saved={isSaved(rec.id)}
            onSave={() => onSave(rec.id)}
          />
        ))}
    </div>
  );
}

function RecCard({
  rec,
  rank,
  saved,
  onSave,
}: {
  rec: Recommendation;
  rank: number;
  saved: boolean;
  onSave: () => void;
}) {
  const pct = Math.round(rec.score * 100);
  const genres = rec.genres
    ?.split(",")
    .slice(0, 2)
    .map((g) => g.trim())
    .join(", ");
  return (
    <article
      className="group animate-in fade-in duration-500"
      style={{ animationDelay: `${rank * 40}ms`, animationFillMode: "backwards" }}
    >
      <div className="relative">
        <MoviePoster title={rec.title} year={rec.year} posterUrl={rec.poster_url} size="fill" />
        <span className="absolute left-2 top-2 rounded-sm bg-background/85 px-1.5 py-0.5 text-[11px] tabular-nums text-foreground/90">
          {rank}
        </span>
        <SaveButton
          saved={saved}
          onClick={onSave}
          title={rec.title}
          className={cn(
            "absolute right-1.5 top-1.5 bg-background/85 hover:bg-background",
            !saved && "opacity-0 group-hover:opacity-100 focus-visible:opacity-100",
          )}
        />
      </div>
      <div className="mt-3">
        <div className="flex items-baseline justify-between gap-2">
          <h3 className="truncate text-sm font-medium">{rec.title}</h3>
          <span className="shrink-0 text-xs tabular-nums text-primary" title="Match score">
            {pct}%
          </span>
        </div>
        <p className="mt-0.5 truncate text-xs text-muted-foreground">
          {[rec.year, genres].filter(Boolean).join(" · ")}
        </p>
        {rec.short && (
          <p className="mt-2 line-clamp-3 text-xs leading-relaxed text-muted-foreground/90">
            {rec.short}
          </p>
        )}
      </div>
    </article>
  );
}
