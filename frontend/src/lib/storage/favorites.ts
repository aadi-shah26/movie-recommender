export interface FavoriteEntry {
  addedAt: number;
  rating: number; // 0-5
  tags: string[];
}

export type FavoritesMap = Record<number, FavoriteEntry>;

export const FAVORITES_KEY = "cinematch:favorites:v1";
export const TAGS_KEY = "cinematch:tags:v1";

export const PRESET_TAGS = ["Watch Later", "Loved", "Rewatch", "Must Watch"];

export function readFavorites(): FavoritesMap {
  if (typeof window === "undefined") return {};
  try {
    const raw = window.localStorage.getItem(FAVORITES_KEY);
    if (!raw) return {};
    return JSON.parse(raw) as FavoritesMap;
  } catch {
    return {};
  }
}

export function writeFavorites(map: FavoritesMap) {
  if (typeof window === "undefined") return;
  window.localStorage.setItem(FAVORITES_KEY, JSON.stringify(map));
}

export function readCustomTags(): string[] {
  if (typeof window === "undefined") return [];
  try {
    const raw = window.localStorage.getItem(TAGS_KEY);
    if (!raw) return [];
    return JSON.parse(raw) as string[];
  } catch {
    return [];
  }
}

export function writeCustomTags(tags: string[]) {
  if (typeof window === "undefined") return;
  window.localStorage.setItem(TAGS_KEY, JSON.stringify(tags));
}
