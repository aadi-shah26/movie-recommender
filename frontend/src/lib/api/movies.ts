// frontend/src/lib/api/movies.ts - talks to backend API
export interface Movie {
  id: number;
  title: string;
  year?: number | null;
  short?: string | null;
  genres?: string | null;
  rating?: number | null;
  certificate?: string | null;
  runtime?: number | null;
  votes?: number | null;
  director?: string | null;
  poster_url?: string | null;
}

export interface Recommendation extends Movie {
  score: number;
}

const API_BASE = (import.meta.env.VITE_API_BASE as string) || "http://localhost:8000";

export async function fetchTitles(): Promise<Movie[]> {
  const res = await fetch(`${API_BASE.replace(/\/$/, "")}/titles`);
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`Failed to fetch titles: ${res.status} ${res.statusText} ${text}`);
  }
  return (await res.json()) as Movie[];
}

export async function fetchRecommendations(liked: number[], k = 5): Promise<Recommendation[]> {
  const res = await fetch(`${API_BASE.replace(/\/$/, "")}/recommendations`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ liked, k }),
  });
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`Failed to fetch recommendations: ${res.status} ${res.statusText} ${text}`);
  }
  return (await res.json()) as Recommendation[];
}
