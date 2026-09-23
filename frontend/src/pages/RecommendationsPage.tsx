import { useEffect, useMemo, useState } from "react";
import { Link, useSearchParams } from "react-router-dom";
import FilterBar, { filtersFromParams, filtersToParams, hasFilters } from "../components/FilterBar";
import {
  api,
  posterUrl,
  type Recommendation,
  type RecommendationsResponse,
  type TasteResponse,
} from "../api";

const ALL = "__all__";

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

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    api
      .recommendations(60, filters)
      .then((r) => {
        if (cancelled) return;
        setRecs(r);
        setError(null);
        setSource(ALL);
      })
      .catch((e) => !cancelled && setError(e instanceof Error ? e.message : String(e)))
      .finally(() => !cancelled && setLoading(false));
    return () => {
      cancelled = true;
    };
  }, [filters]);

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
  if (!recs) return <p className="text-zinc-500">Loading…</p>;
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
          Ranked by the average of your taste profile ① and embedding similarity ②
          {taste?.n_rated ? ` (built from ${taste.n_rated} rated films` : ""}
          {taste?.clusters.length ? `, ${taste.clusters.length} taste clusters)` : taste?.n_rated ? ")" : ""}.
          The collaborative score and learned blend arrive in later milestones.
        </p>
      </div>

      <FilterBar
        filters={filters}
        facets={recs.facets}
        onChange={(next) => setParams(filtersToParams(next), { replace: true })}
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
            <RecCard key={r.tmdb_id} rec={r} rank={recs.items.indexOf(r) + 1 || i + 1} />
          ))}
        </div>
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

function RecCard({ rec, rank }: { rec: Recommendation; rank: number }) {
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
          {rec.vote_average !== null && (
            <p
              className="mt-0.5 text-xs text-amber-300/90"
              title={rec.vote_count ? `TMDB user score from ${rec.vote_count.toLocaleString()} votes` : "TMDB user score"}
            >
              ★ {rec.vote_average.toFixed(1)}
              <span className="text-zinc-500"> TMDB</span>
            </p>
          )}
        </div>
        {rec.profile_score !== null && (
          <ScoreBar
            label="① taste profile"
            value={rec.profile_score}
            detail={rec.profile_raw !== null ? `raw ${rec.profile_raw.toFixed(2)}` : undefined}
          />
        )}
        <ScoreBar label="② similarity" value={rec.embedding_score} detail={`cos ${rec.similarity.toFixed(2)}`} />
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
