export interface Health {
  status: string;
  features: { tmdb: boolean; omdb: boolean; openai: boolean };
}

export interface UploadResult {
  run_id: number;
  files_found: string[];
  stats: Record<string, number>;
  warnings: string[];
}

export interface IngestRun {
  id: number;
  started_at: string;
  finished_at: string | null;
  status: "running" | "done" | "error";
  stage: string;
  progress_done: number;
  progress_total: number;
  message: string | null;
  stats: {
    added?: number;
    changed?: number;
    unchanged?: number;
    removed?: number;
    films?: number;
    rated?: number;
    watchlist?: number;
    warnings?: number;
    matching?: Record<string, number>;
    enrichment?: Record<string, number>;
  };
}

export interface RunOut {
  run: IngestRun;
  stages: string[];
}

export interface MovieBrief {
  tmdb_id: number;
  title: string;
  year: number | null;
  poster_path: string | null;
  directors: string[];
  overview: string | null;
}

export type MatchStatus = "matched" | "low_confidence" | "unmatched" | "manual" | "error" | "ignored";

export interface MatchRow {
  film_key: string;
  name: string;
  year: number | null;
  rating: number | null;
  status: MatchStatus | null;
  confidence: number | null;
  note: string | null;
  tmdb_id: number | null;
  movie: MovieBrief | null;
}

export interface MatchesResponse {
  counts: Record<string, number>;
  rows: MatchRow[];
}

export interface Recommendation {
  tmdb_id: number;
  title: string;
  year: number | null;
  directors: string[];
  genres: string[];
  poster_path: string | null;
  overview: string | null;
  runtime: number | null;
  vote_average: number | null;
  vote_count: number | null;
  original_language: string | null;
  in_watchlist: boolean;
  score: number;
  embedding_score: number;
  similarity: number;
  source: string;
  source_label: string;
  candidate_sources: string[];
  profile_score: number | null;
  profile_raw: number | null;
  profile_reasons: ProfileReason[];
}

/** One feature behind score ①, e.g. "Director Denis Villeneuve", +0.9★ over 4 films. */
export interface ProfileReason {
  type: string;
  value: string;
  label: string;
  stars: number;
  n: number;
  contribution: number;
}

export interface Facets {
  genres: Record<string, number>;
  decades: Record<string, number>;
  languages: Record<string, number>;
}

export interface RecommendationsResponse {
  ready: boolean;
  message: string | null;
  items: Recommendation[];
  total?: number;
  matching?: number;
  facets?: Facets;
}

export interface RecFilters {
  min_rating?: number;
  genre?: string;
  decade?: number;
  max_runtime?: number;
  language?: string;
}

export interface TasteCluster {
  id: number;
  source: string;
  label: string;
  size: number;
  examples: { tmdb_id: number; title: string; year: number | null; poster_path: string | null }[];
}

export interface TasteResponse {
  ready: boolean;
  message?: string | null;
  mean_rating?: number;
  n_rated?: number;
  silhouette?: number | null;
  embedding_model?: string;
  clusters: TasteCluster[];
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const resp = await fetch(path, init);
  if (!resp.ok) {
    let detail = resp.statusText;
    try {
      detail = (await resp.json()).detail ?? detail;
    } catch {
      /* non-JSON error body */
    }
    throw new Error(detail);
  }
  return resp.json() as Promise<T>;
}

const postJson = <T>(path: string, body: unknown) =>
  request<T>(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });

export const api = {
  health: () => request<Health>("/api/health"),
  upload: (file: File) => {
    const body = new FormData();
    body.append("file", file);
    return request<UploadResult>("/api/upload", { method: "POST", body });
  },
  run: (id: number) => request<RunOut>(`/api/ingest/${id}`),
  latestRun: () => request<RunOut | null>("/api/ingest/latest"),
  resume: () => request<{ run_id: number }>("/api/ingest/resume", { method: "POST" }),
  matches: (filter: "review" | "all") => request<MatchesResponse>(`/api/matches?filter=${filter}`),
  setMatch: (film_key: string, tmdb_ref: string) =>
    postJson<MatchRow>("/api/matches/set", { film_key, tmdb_ref }),
  acceptMatch: (film_key: string) => postJson<MatchRow>("/api/matches/accept", { film_key }),
  ignoreMatch: (film_key: string) => postJson<MatchRow>("/api/matches/ignore", { film_key }),
  recommendations: (limit = 60, filters: RecFilters = {}) => {
    const qs = new URLSearchParams({ limit: String(limit) });
    for (const [k, v] of Object.entries(filters)) {
      if (v !== undefined && v !== "") qs.set(k, String(v));
    }
    return request<RecommendationsResponse>(`/api/recommendations?${qs}`);
  },
  taste: () => request<TasteResponse>("/api/taste"),
};

export const posterUrl = (path: string | null, size: "w92" | "w185" | "w342" = "w92") =>
  path ? `https://image.tmdb.org/t/p/${size}${path}` : null;
