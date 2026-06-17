import { MOCK_MOVIES, mockRecommendations } from "./mock-movies";

export interface Movie {
  id: number;
  title: string;
  year?: number | null;
  short?: string | null;
}

export interface Recommendation extends Movie {
  score: number;
}

export async function fetchTitles(): Promise<Movie[]> {
  await new Promise((r) => setTimeout(r, 400));
  return MOCK_MOVIES;
}

export async function fetchRecommendations(
  liked: number[],
  k = 5,
): Promise<Recommendation[]> {
  await new Promise((r) => setTimeout(r, 700));
  return mockRecommendations(liked, k);
}
