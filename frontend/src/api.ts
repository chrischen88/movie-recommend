export interface Health {
  status: string;
  features: { tmdb: boolean; omdb: boolean; openai: boolean };
}

export interface UploadResult {
  session_id: string;
  expires_in: number;
  files_found: string[];
  stats: Record<string, number>;
  warnings: string[];
}

export interface Run {
  id: number;
  started_at: string;
  finished_at: string | null;
  status: "queued" | "running" | "done" | "error" | "cancelled";
  stage: string;
  progress_done: number;
  progress_total: number;
  message: string | null;
  stats: {
    films?: number;
    rated?: number;
    watchlist?: number;
    warnings?: number;
    /** Match fixes saved in this browser that applied to this export. */
    fixes_applied?: number;
    matching?: Record<string, number>;
    enrichment?: Record<string, number>;
  };
}

export interface RunOut {
  run: Run;
  stages: string[];
  /** 0 while processing, n while n exports are ahead in line, null when idle. */
  queue_position: number | null;
  /** Seconds until the session is deleted if unused. */
  expires_in: number;
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
  collab_score: number | null;
  collab_predicted: number | null;
  /** The learned blend's predicted rating (stars); null when fixed weights are in use. */
  predicted_rating: number | null;
  imdb_rating: number | null;
  rt_score: number | null;
  metacritic: number | null;
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
  /** Your rated films found in MovieLens; null when there's no collaborative score. */
  collab_films?: number | null;
  blend_mode?: "learned" | "fixed";
}

export type RatingSource = "tmdb" | "imdb" | "rt" | "metacritic";

export interface RecFilters {
  min_rating?: number;
  rating_source?: RatingSource;
  hide_low_quality?: boolean;
  genre?: string;
  decade?: number;
  max_runtime?: number;
  language?: string;
}

export interface MethodMetrics {
  /** null for "ranking", which orders films rather than predicting stars. */
  rmse: number | null;
  spearman: number | null;
  n: number;
}

export interface LinearBlend {
  features: string[];
  coef: number[];
  intercept: number;
  alpha: number;
}

export interface MetricsResponse {
  ready: boolean;
  message?: string;
  mode?: "learned" | "fixed";
  min_ratings_for_learning?: number;
  n_ratings?: number;
  n_with_collab?: number;
  trained_at?: string;
  fixed_weights?: Record<string, number>;
  /** Share of taste fit (①②) in the learned-mode ranking. */
  rank_fit_weight?: number;
  full?: LinearBlend | null;
  partial?: LinearBlend | null;
  metrics?: Record<string, MethodMetrics>;
  collab_model?: { dataset: string; val_rmse: number; val_rmse_global_mean: number; n_users: number; n_items: number } | null;
  omdb?: { enabled: boolean; used_today: number; daily_limit: number };
  candidates?: { total: number; by_source: Record<string, number> };
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

// ---------------------------------------------------------------- session & saved fixes

// Your data lives in server memory for one visit; this id is its key. The match
// fixes you make are kept here in the browser and sent with every upload.
const SESSION_KEY = "sessionId";
const FIXES_KEY = "matchFixes";

export type MatchFix = { tmdb_id: number } | { ignored: true };

function storageGet(key: string): string | null {
  try {
    return window.localStorage.getItem(key);
  } catch {
    return null;
  }
}

function storageSet(key: string, value: string | null): void {
  try {
    if (value === null) window.localStorage.removeItem(key);
    else window.localStorage.setItem(key, value);
  } catch {
    /* storage unavailable (private mode): the session lasts for this page only */
  }
}

let memorySession: string | null = null;
const sessionId = () => storageGet(SESSION_KEY) ?? memorySession;
function setSessionId(id: string | null) {
  memorySession = id;
  storageSet(SESSION_KEY, id);
}

export const hasSession = () => sessionId() !== null;

/** Why there's nothing to show: never uploaded, or the session is gone. */
let sessionEnded = false;
export const noSessionMessage = () =>
  sessionEnded
    ? "Your session ended (it's deleted after an hour without use). Upload your export again."
    : "Upload your Letterboxd export, or try the sample profile, to get started.";

export function savedFixes(): Record<string, MatchFix> {
  try {
    const parsed = JSON.parse(storageGet(FIXES_KEY) ?? "{}");
    return parsed && typeof parsed === "object" && !Array.isArray(parsed) ? parsed : {};
  } catch {
    return {};
  }
}

function saveFix(filmKey: string, fix: MatchFix) {
  storageSet(FIXES_KEY, JSON.stringify({ ...savedFixes(), [filmKey]: fix }));
}

export const clearSavedFixes = () => storageSet(FIXES_KEY, null);

class SessionGone extends Error {}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const headers = new Headers(init?.headers);
  const id = sessionId();
  if (id) headers.set("X-Session-Id", id);
  const resp = await fetch(path, { ...init, headers });
  if (resp.status === 410) {
    setSessionId(null);
    sessionEnded = true;
    throw new SessionGone("session ended");
  }
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

/** Calls that need a session answer `fallback` when there isn't one. */
async function withSession<T>(call: () => Promise<T>, fallback: () => T): Promise<T> {
  if (!hasSession()) return fallback();
  try {
    return await call();
  } catch (e) {
    if (e instanceof SessionGone) return fallback();
    throw e;
  }
}

const postJson = <T>(path: string, body: unknown) =>
  request<T>(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });

const saving = (row: Promise<MatchRow>, fix: (r: MatchRow) => MatchFix | null) =>
  row.then((r) => {
    const f = fix(r);
    if (f) saveFix(r.film_key, f);
    return r;
  });

async function startSession(upload: Promise<UploadResult>): Promise<UploadResult> {
  const result = await upload;
  setSessionId(result.session_id);
  sessionEnded = false;
  return result;
}

export const api = {
  health: () => request<Health>("/api/health"),
  upload: async (file: File) => {
    const body = new FormData();
    body.append("file", file);
    body.append("fixes", JSON.stringify(savedFixes()));
    return startSession(request<UploadResult>("/api/sessions", { method: "POST", body }));
  },
  /** A session on the built-in sample profile, for visitors without an export. */
  demo: () => startSession(request<UploadResult>("/api/sessions/demo", { method: "POST" })),
  session: () => withSession<RunOut | null>(() => request<RunOut>("/api/session"), () => null),
  reprocess: () => request<RunOut>("/api/session/reprocess", { method: "POST" }),
  forget: async () => {
    try {
      await request<{ deleted: boolean }>("/api/session", { method: "DELETE" });
    } finally {
      setSessionId(null);
      sessionEnded = false;
    }
  },
  matches: (filter: "review" | "all") =>
    withSession<MatchesResponse>(
      () => request<MatchesResponse>(`/api/matches?filter=${filter}`),
      () => ({ counts: {}, rows: [] }),
    ),
  setMatch: (film_key: string, tmdb_ref: string) =>
    saving(postJson<MatchRow>("/api/matches/set", { film_key, tmdb_ref }), (r) =>
      r.tmdb_id ? { tmdb_id: r.tmdb_id } : null,
    ),
  acceptMatch: (film_key: string) =>
    saving(postJson<MatchRow>("/api/matches/accept", { film_key }), (r) =>
      r.tmdb_id ? { tmdb_id: r.tmdb_id } : null,
    ),
  ignoreMatch: (film_key: string) =>
    saving(postJson<MatchRow>("/api/matches/ignore", { film_key }), () => ({ ignored: true })),
  recommendations: (limit = 60, filters: RecFilters = {}) => {
    const qs = new URLSearchParams({ limit: String(limit) });
    for (const [k, v] of Object.entries(filters)) {
      if (v !== undefined && v !== "") qs.set(k, String(v));
    }
    return withSession<RecommendationsResponse>(
      () => request<RecommendationsResponse>(`/api/recommendations?${qs}`),
      () => ({ ready: false, message: noSessionMessage(), items: [] }),
    );
  },
  taste: () =>
    withSession<TasteResponse>(
      () => request<TasteResponse>("/api/taste"),
      () => ({ ready: false, message: noSessionMessage(), clusters: [] }),
    ),
  metrics: () =>
    withSession<MetricsResponse>(
      () => request<MetricsResponse>("/api/metrics"),
      () => ({ ready: false, message: noSessionMessage() }),
    ),
};

export const posterUrl = (path: string | null, size: "w92" | "w185" | "w342" = "w92") =>
  path ? `https://image.tmdb.org/t/p/${size}${path}` : null;
