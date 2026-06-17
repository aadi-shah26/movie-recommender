## Goal

Add a **Favorites / Watchlist** section to CineMatch plus four upgrades: custom tags, movie posters, star ratings, and advanced filters. Everything stays client-side (localStorage) and lives on the same page.

## New layout

The page becomes a 3-column responsive grid (stacks on mobile, 2-col on tablet, 3-col on desktop):

```text
┌──────────────┬──────────────┬──────────────┐
│  Library     │ Recommended  │  Favorites   │
│  (browse +   │  (results)   │  (saved for  │
│   filter)    │              │   later)     │
└──────────────┴──────────────┴──────────────┘
```

A **filter bar** sits above the grid (decade, min year, sort). Each movie card gains a **poster, star-rating control, heart/favorite button, and tag chips**.

## Features

### 1. Favorites panel (third column)
- Heart icon on every `MovieCard` and `RecCard` toggles favorite state.
- Favorites panel lists saved movies with poster, title, year, rating, tags, and remove button.
- Empty state: "No favorites yet — tap the ♥ on any movie to save it for later."
- Header shows count badge + "Clear all" action.

### 2. Custom tags
- Built-in presets: `Watch Later`, `Loved`, `Rewatch`, `Must Watch`.
- Tag picker on each favorited movie (small popover with checkbox list + free-text input to add a custom tag).
- Tag chips render under the movie title in the favorites panel.
- Filter favorites by tag via chip row at the top of the favorites panel.

### 3. Star ratings (1–5)
- 5-star control on favorite cards (hover to preview, click to set, click again to clear).
- Rating displayed in the favorites panel; sort option "By rating".

### 4. Movie posters
- No external API required: generate **deterministic gradient posters** per movie (hash of title → two-color gradient + large title initials overlay). Looks polished and works fully offline with mock data.
- Poster appears on library cards (small), recommendation cards (medium), and favorites cards (medium).
- Component: `<MoviePoster movie={m} size="sm|md" />`.

### 5. Advanced filters (above the library)
- **Decade** dropdown (All / 1990s / 2000s / 2010s / 2020s).
- **Min match score** slider (only affects the Recommendations column).
- **Sort** for library: Title (A–Z), Year (new→old), Year (old→new).
- Active filters render as removable chips.

### 6. Persistence
- New `src/lib/storage/favorites.ts`:
  - Shape: `{ [movieId: number]: { addedAt: number; rating: number; tags: string[] } }`.
  - Stored under key `cinematch:favorites:v1` in `localStorage`.
  - Custom tag vocabulary stored under `cinematch:tags:v1`.
- New `useFavorites()` hook wrapping `useState` + `useEffect` for read/write/sync across tabs (storage event listener).

## Technical details

- New files:
  - `src/lib/storage/favorites.ts` — typed read/write helpers.
  - `src/hooks/use-favorites.ts` — React hook exposing `favorites`, `toggle`, `setRating`, `setTags`, `clearAll`, `isFavorite`.
  - `src/components/movie-poster.tsx` — deterministic gradient + initials poster.
  - `src/components/star-rating.tsx` — 5-star input.
  - `src/components/tag-picker.tsx` — popover with preset + custom tag entry (uses existing shadcn `Popover`, `Checkbox`, `Input`).
  - `src/components/favorites-panel.tsx` — third column UI.
  - `src/components/filter-bar.tsx` — decade / sort / min-score controls.
- Modified files:
  - `src/routes/index.tsx` — switch to 3-column grid, integrate hook, wire filter state, pass props to existing cards.
- No backend / API / route changes. Mock data stays.
- No new npm packages — uses existing shadcn primitives (`Popover`, `Slider`, `Select`, `Checkbox`) and `lucide-react` icons (`Heart`, `Tag`, `Filter`, `Star`).

## Out of scope (intentionally)

- Real movie poster images (would need TMDB API key + secret). Happy to add this in a follow-up if you want — say the word and I'll wire up TMDB.
- Multi-device sync (requires backend/auth).
