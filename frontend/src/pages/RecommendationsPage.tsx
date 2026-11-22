import { useEffect, useMemo, useState } from "react";
import { Link, useSearchParams } from "react-router-dom";
import FilterBar, { filtersFromParams, filtersToParams, hasFilters } from "../components/FilterBar";
import {
  api,
  posterUrl,
  type Recommendation,
  type RecommendationsResponse,
  type RequestInterpretation,
  type TasteResponse,
} from "../api";

const ALL = "__all__";

const AI_WAIT_MS = 15000;

export default function RecommendationsPage() {
  const [recs, setRecs] = useState<RecommendationsResponse | null>(null);
  const [taste, setTaste] = useState<TasteResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [source, setSource] = useState<string>(ALL);
  const [loading, setLoading] = useState(true);
  const [params, setParams] = useSearchParams();
  const filters = useMemo(() => filtersFromParams(params), [params]);

  useEffect(() => {
    api.taste().then(setTaste).catch(() => {});
  }, []);

  const q = params.get("q") ?? "";
  const [draft, setDraft] = useState(q);
  const [refining, setRefining] = useState(false);
  // AI picks that arrived after the standard ranking was already shown: offered
  // with a button, never swapped in under the reader.
  const [aiReady, setAiReady] = useState<RecommendationsResponse | null>(null);
  useEffect(() => setDraft(q), [q]);

  const setQuery = (text: string) => {
    const next = filtersToParams(filters);
    if (text.trim()) next.set("q", text.trim());
    setParams(next, { replace: true });
  };

  // When AI is available, wait for its picks (usually instant: they're prepared
  // while the export is processed) instead of showing a list that then reorders.
  // If they take longer than AI_WAIT_MS, show the standard ranking and offer the
  // AI picks with a button when they arrive.
  // `cancelled` + abort make sure a slow response for old inputs never lands.
  useEffect(() => {
    let cancelled = false;
    let fallbackShown = false;
    let fallback: number | undefined;
    const ctl = new AbortController();
    const opts = { q, signal: ctl.signal };
    setLoading(true);
    setRefining(false);
    setAiReady(null);
    const show = (r: RecommendationsResponse) => {
      setRecs(r);
      setError(null);
      setSource(ALL);
      setLoading(false);
    };
    api
      .recommendations(60, filters, { ...opts, rerank: false })
      .then((r) => {
        if (cancelled) return;
        if (!r.ready || !r.llm?.available) return show(r);
        setRefining(true);
        fallback = window.setTimeout(() => {
          if (cancelled) return;
          fallbackShown = true;
          show(r);
        }, AI_WAIT_MS);
        api
          .recommendations(60, filters, { ...opts, rerank: true })
          .then((r2) => {
            if (cancelled) return;
            const best = r2.ready ? r2 : r;
            if (fallbackShown && best.llm?.reranked) setAiReady(best);
            else if (!fallbackShown) show(best);
          })
          .catch(() => !cancelled && !fallbackShown && show(r))
          .finally(() => {
            window.clearTimeout(fallback);
            if (!cancelled) setRefining(false);
          });
      })
      .catch((e) => {
        if (cancelled) return;
        setError(e instanceof Error ? e.message : String(e));
        setLoading(false);
      });
    return () => {
      cancelled = true;
      window.clearTimeout(fallback);
      ctl.abort();
    };
  }, [filters, q]);

  const sources = useMemo(() => {
    const counts = new Map<string, { label: string; n: number }>();
    for (const r of recs?.items ?? []) {
      const cur = counts.get(r.source) ?? { label: r.source_label, n: 0 };
      counts.set(r.source, { ...cur, n: cur.n + 1 });
    }
    return [...counts.entries()].sort((a, b) => b[1].n - a[1].n);
  }, [recs]);

  const shown = (recs?.items ?? []).filter((r) => source === ALL || r.source === source);

  if (error) {
    return <div className="rounded-md bg-red-950/50 border border-red-800 p-3 text-red-300">{error}</div>;
  }
  if (!recs) {
    return <p className={refining ? "animate-pulse text-zinc-400" : "text-zinc-500"}>
      {refining ? "AI is picking and explaining your films…" : "Loading…"}
    </p>;
  }
  if (!recs.ready) {
    return (
      <div className="rounded-xl border border-zinc-800 p-8 text-center space-y-3">
        <p className="text-zinc-300">{recs.message ?? "Recommendations aren't ready yet."}</p>
        <Link to="/" className="inline-block rounded-md bg-emerald-600 px-4 py-2 text-sm hover:bg-emerald-500">
          Upload your export
        </Link>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold">Recommendations</h1>
        <p className="text-zinc-400 text-sm mt-1">
          {recs.blend_mode === "learned"
            ? "Ranked by a blend of your taste profile ①, embedding similarity ②"
            : "Ranked by fixed weights on your taste profile ①, embedding similarity ②"}
          {recs.collab_films != null ? " and collaborative score ③" : ""}
          {recs.blend_mode === "learned"
            ? ": your predicted rating (weighted by how well each score predicted your own ratings) mixed with how well it fits your taste (① and ②)"
            : ` (too few ratings to learn weights yet)`}
          {taste?.n_rated ? `. Built from ${taste.n_rated} rated films` : ""}
          {taste?.clusters.length ? ` in ${taste.clusters.length} taste clusters` : ""}.
          {recs.collab_films != null
            ? ` ③ compares ${recs.collab_films} of your ratings with MovieLens users; films not in MovieLens are ranked on ① and ② only.`
            : " The collaborative score ③ appears once the MovieLens model is trained (make train)."}{" "}
          <Link to="/metrics" className="text-emerald-400 hover:underline">
            How well does it work?
          </Link>
        </p>
      </div>

      <form
        className="flex gap-2"
        onSubmit={(e) => {
          e.preventDefault();
          setQuery(draft);
        }}
      >
        <div className="relative min-w-0 flex-1">
          <input
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            maxLength={300}
            placeholder={'Ask for something: "a tense 90-minute thriller", "like Arrival but lighter"'}
            className="w-full rounded-md border border-zinc-700 bg-zinc-900 py-2 pl-3 pr-8 text-sm placeholder:text-zinc-600 focus:border-emerald-500 focus:outline-none"
          />
          {q && (
            <button
              type="button"
              aria-label="Clear request"
              title="Clear request"
              onClick={() => setQuery("")}
              className="absolute right-1 top-1/2 -translate-y-1/2 px-2 text-lg leading-none text-zinc-500 hover:text-white"
            >
              ×
            </button>
          )}
        </div>
        <button
          type="submit"
          disabled={!draft.trim() || draft.trim() === q}
          className="shrink-0 rounded-md bg-emerald-600 px-4 py-2 text-sm hover:bg-emerald-500 disabled:opacity-40 disabled:hover:bg-emerald-600"
        >
          Ask
        </button>
      </form>

      {recs.request && <Interpreted request={recs.request} />}

      {(refining || aiReady || recs.llm?.reranked || recs.llm?.message) && (
        <p className="flex flex-wrap items-center gap-2 text-xs text-zinc-500" aria-live="polite">
          {refining && loading && (
            <span className="animate-pulse">AI is picking and explaining films for these settings…</span>
          )}
          {refining && !loading && (
            <span className="animate-pulse">
              Showing the standard ranking while AI picks and explains (you'll be asked before the list changes)…
            </span>
          )}
          {aiReady && (
            <>
              <span>AI picks are ready.</span>
              <button
                onClick={() => {
                  setRecs(aiReady);
                  setAiReady(null);
                  setSource(ALL);
                }}
                className="rounded bg-emerald-700 px-2 py-0.5 text-white hover:bg-emerald-600"
              >
                Show AI picks
              </button>
            </>
          )}
          {!loading && recs.llm?.reranked && (
            <span className="rounded bg-emerald-950/60 px-1.5 py-0.5 text-emerald-300" title="Ordered and explained by AI">
              AI-ranked
            </span>
          )}
          {!loading && !refining && !aiReady && !recs.llm?.reranked && recs.llm?.message && (
            <span>Standard ranking ({recs.llm.message})</span>
          )}
        </p>
      )}

      <FilterBar
        filters={filters}
        facets={recs.facets}
        onChange={(next) => {
          const p = filtersToParams(next);
          if (q) p.set("q", q);
          setParams(p, { replace: true });
        }}
      />

      {recs.total !== undefined && hasFilters(filters) && (
        <p className="text-sm text-zinc-400">
          {recs.matching} of {recs.total} candidates match your filters
          {recs.matching! > recs.items.length && ` · showing the top ${recs.items.length}`}
        </p>
      )}

      {sources.length > 1 && (
        <div className="flex flex-wrap gap-2 text-sm">
          <Chip active={source === ALL} onClick={() => setSource(ALL)}>
            All ({recs.items.length})
          </Chip>
          {sources.map(([key, { label, n }]) => (
            <Chip key={key} active={source === key} onClick={() => setSource(key)}>
              {label} ({n})
            </Chip>
          ))}
        </div>
      )}

      {shown.length === 0 ? (
        <p className="rounded-xl border border-zinc-800 p-8 text-center text-zinc-400">
          {hasFilters(filters) ? "No films match these filters. Try loosening one." : "No unseen candidates yet."}
        </p>
      ) : (
        <div
          className={`grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-5 transition-opacity ${loading ? "opacity-50" : ""}`}
        >
          {shown.map((r, i) => (
            <RecCard key={r.tmdb_id} rec={r} rank={recs.items.indexOf(r) + 1 || i + 1} collabOn={recs.collab_films != null} />
          ))}
        </div>
      )}
    </div>
  );
}

function Interpreted({ request: r }: { request: RequestInterpretation }) {
  const chips = [
    ...r.genres,
    ...(r.decade ? [`${r.decade}s`] : []),
    ...(r.max_runtime ? [`≤ ${r.max_runtime} min`] : []),
    ...(r.language ? [r.language] : []),
    ...r.keywords,
  ];
  return (
    <div className="flex flex-wrap items-center gap-1.5 text-xs">
      {r.by_llm ? (
        <>
          <span className="text-zinc-500">Interpreted as:</span>
          {chips.map((c, i) => (
            <span key={`${c}:${i}`} className="rounded-full border border-zinc-700 px-2 py-0.5 text-zinc-300">
              {c}
            </span>
          ))}
          {chips.length === 0 && <span className="text-zinc-500">no specific filters</span>}
        </>
      ) : (
        <span className="text-zinc-500">
          Searched your words directly{r.note ? ` (AI unavailable: ${r.note})` : ""}
        </span>
      )}
    </div>
  );
}

function Chip({
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
      onClick={onClick}
      className={`rounded-full px-3 py-1 border ${
        active ? "border-emerald-500 bg-emerald-950/50 text-emerald-200" : "border-zinc-700 text-zinc-400 hover:text-white"
      }`}
    >
      {children}
    </button>
  );
}

function RecCard({ rec, rank, collabOn }: { rec: Recommendation; rank: number; collabOn: boolean }) {
  const poster = posterUrl(rec.poster_path, "w342");
  return (
    <article className="group flex flex-col overflow-hidden rounded-xl border border-zinc-800 bg-zinc-900/40">
      <a
        href={`https://www.themoviedb.org/movie/${rec.tmdb_id}`}
        target="_blank"
        rel="noreferrer"
        className="relative aspect-[2/3] bg-zinc-800"
      >
        {poster ? (
          <img src={poster} alt="" loading="lazy" className="h-full w-full object-cover" />
        ) : (
          <div className="grid h-full place-items-center p-4 text-center text-zinc-500">{rec.title}</div>
        )}
        <span className="absolute left-2 top-2 rounded bg-black/70 px-1.5 py-0.5 text-xs tabular-nums">#{rank}</span>
        {rec.in_watchlist && (
          <span className="absolute right-2 top-2 rounded bg-sky-700/90 px-1.5 py-0.5 text-xs">watchlist</span>
        )}
      </a>
      <div className="flex flex-1 flex-col gap-2 p-3">
        <div>
          <h2 className="font-medium leading-tight">{rec.title}</h2>
          <p className="text-xs text-zinc-500">
            {[rec.year, rec.directors.slice(0, 2).join(", "), rec.runtime ? `${rec.runtime} min` : null]
              .filter(Boolean)
              .join(" · ")}
          </p>
          {rec.predicted_rating !== null && (
            <p
              className="mt-1 text-sm font-medium text-emerald-300"
              title="The learned blend's estimate of how you'd rate it"
            >
              ~{rec.predicted_rating.toFixed(1)}★ for you
            </p>
          )}
          <Ratings rec={rec} />
        </div>
        {rec.why && (
          <p className="text-xs italic text-zinc-300">
            {rec.why_source === "llm" && (
              <span className="mr-1 rounded bg-emerald-950/60 px-1 text-[10px] not-italic text-emerald-300">AI</span>
            )}
            {rec.why}
          </p>
        )}
        {rec.profile_score !== null && (
          <ScoreBar
            label="① taste profile"
            value={rec.profile_score}
            detail={rec.profile_raw !== null ? `raw ${rec.profile_raw.toFixed(2)}` : undefined}
          />
        )}
        <ScoreBar label="② similarity" value={rec.embedding_score} detail={`cos ${rec.similarity.toFixed(2)}`} />
        {rec.collab_score !== null && rec.collab_predicted !== null ? (
          <ScoreBar
            label={`③ others like you · ${rec.collab_predicted.toFixed(1)}★`}
            value={rec.collab_score}
            detail={`Predicted rating ${rec.collab_predicted.toFixed(2)}★ from MovieLens users with similar ratings`}
          />
        ) : collabOn && (
          <p className="text-[11px] text-zinc-600" title="Not in MovieLens (often too recent), so ranked on ① and ② only">
            ③ not in MovieLens
          </p>
        )}
        {rec.profile_reasons.length > 0 && (
          <ul className="space-y-0.5 text-xs">
            {rec.profile_reasons.slice(0, 3).map((r) => (
              <li
                key={`${r.type}:${r.value}`}
                className="flex justify-between gap-2"
                title={`You rate films with this ${r.stars >= 0 ? "above" : "below"} your average (${r.n} rated film${r.n === 1 ? "" : "s"}, shrunk toward 0)`}
              >
                <span className="truncate text-zinc-400">{r.label}</span>
                <span className={`tabular-nums ${r.stars >= 0 ? "text-emerald-400" : "text-rose-400"}`}>
                  {r.stars >= 0 ? "+" : "−"}
                  {Math.abs(r.stars).toFixed(1)}★
                </span>
              </li>
            ))}
          </ul>
        )}
        <p className="mt-auto text-xs text-zinc-400">
          <span className="text-zinc-500">Matches:</span> {rec.source_label}
        </p>
      </div>
    </article>
  );
}

function Ratings({ rec }: { rec: Recommendation }) {
  const items: { label: string; value: string; title: string }[] = [];
  if (rec.vote_average !== null) {
    items.push({
      label: "TMDB",
      value: rec.vote_average.toFixed(1),
      title: rec.vote_count ? `TMDB user score from ${rec.vote_count.toLocaleString()} votes` : "TMDB user score",
    });
  }
  if (rec.imdb_rating !== null) items.push({ label: "IMDb", value: rec.imdb_rating.toFixed(1), title: "IMDb rating" });
  if (rec.rt_score !== null) items.push({ label: "RT", value: `${rec.rt_score}%`, title: "Rotten Tomatoes Tomatometer" });
  if (rec.metacritic !== null) items.push({ label: "MC", value: String(rec.metacritic), title: "Metacritic Metascore" });
  if (!items.length) return null;
  return (
    <p className="mt-0.5 flex flex-wrap gap-x-2 text-xs text-amber-300/90">
      {items.map((it) => (
        <span key={it.label} title={it.title} className="tabular-nums">
          {it.value}
          <span className="text-zinc-500"> {it.label}</span>
        </span>
      ))}
    </p>
  );
}

function ScoreBar({ label, value, detail }: { label: string; value: number; detail?: string }) {
  return (
    <div title={detail}>
      <div className="flex justify-between text-[11px] text-zinc-500">
        <span>{label}</span>
        <span className="tabular-nums">{Math.round(value * 100)}</span>
      </div>
      <div className="mt-0.5 h-1 rounded bg-zinc-800">
        <div className="h-full rounded bg-emerald-500" style={{ width: `${value * 100}%` }} />
      </div>
    </div>
  );
}
