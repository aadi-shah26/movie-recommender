import { useCallback, useEffect, useState } from "react";
import {
  FAVORITES_KEY,
  TAGS_KEY,
  type FavoritesMap,
  readCustomTags,
  readFavorites,
  writeCustomTags,
  writeFavorites,
} from "@/lib/storage/favorites";

export function useFavorites() {
  const [favorites, setFavorites] = useState<FavoritesMap>({});
  const [customTags, setCustomTags] = useState<string[]>([]);
  const [hydrated, setHydrated] = useState(false);

  useEffect(() => {
    setFavorites(readFavorites());
    setCustomTags(readCustomTags());
    setHydrated(true);

    const onStorage = (e: StorageEvent) => {
      if (e.key === FAVORITES_KEY) setFavorites(readFavorites());
      if (e.key === TAGS_KEY) setCustomTags(readCustomTags());
    };
    window.addEventListener("storage", onStorage);
    return () => window.removeEventListener("storage", onStorage);
  }, []);

  const persist = useCallback((next: FavoritesMap) => {
    setFavorites(next);
    writeFavorites(next);
  }, []);

  const toggle = useCallback(
    (id: number) => {
      const next = { ...favorites };
      if (next[id]) {
        delete next[id];
      } else {
        next[id] = { addedAt: Date.now(), rating: 0, tags: [] };
      }
      persist(next);
    },
    [favorites, persist],
  );

  const setRating = useCallback(
    (id: number, rating: number) => {
      if (!favorites[id]) return;
      persist({ ...favorites, [id]: { ...favorites[id], rating } });
    },
    [favorites, persist],
  );

  const setTags = useCallback(
    (id: number, tags: string[]) => {
      if (!favorites[id]) return;
      persist({ ...favorites, [id]: { ...favorites[id], tags } });
    },
    [favorites, persist],
  );

  const clearAll = useCallback(() => persist({}), [persist]);

  const isFavorite = useCallback((id: number) => Boolean(favorites[id]), [favorites]);

  const addCustomTag = useCallback(
    (tag: string) => {
      const trimmed = tag.trim();
      if (!trimmed) return;
      if (customTags.includes(trimmed)) return;
      const next = [...customTags, trimmed];
      setCustomTags(next);
      writeCustomTags(next);
    },
    [customTags],
  );

  return {
    favorites,
    customTags,
    hydrated,
    toggle,
    setRating,
    setTags,
    clearAll,
    isFavorite,
    addCustomTag,
  };
}
