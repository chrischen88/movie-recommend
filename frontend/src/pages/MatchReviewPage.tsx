import { useCallback, useEffect, useState } from "react";
import { api, posterUrl, type MatchRow, type MatchStatus } from "../api";

type Filter = "review" | "all";

const STATUS_STYLE: Record<MatchStatus, string> = {
  matched: "bg-emerald-900/60 text-emerald-300",
  manual: "bg-sky-900/60 text-sky-300",
  low_confidence: "bg-amber-900/60 text-amber-300",
  unmatched: "bg-red-900/60 text-red-300",
  error: "bg-red-900/60 text-red-300",
  ignored: "bg-zinc-800 text-zinc-400",
};

const STATUS_LABEL: Record<MatchStatus, string> = {
  matched: "matched",
  manual: "confirmed",
  low_confidence: "low confidence",
  unmatched: "no match",
  error: "error",
  ignored: "ignored",
};

export default function MatchReviewPage() {
  const [filter, setFilter] = useState<Filter>("review");
  const [rows, setRows] = useState<MatchRow[] | null>(null);
  const [counts, setCounts] = useState<Record<string, number>>({});
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    try {
      const res = await api.matches(filter);
      setRows(res.rows);
      setCounts(res.counts);
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }, [filter]);

  useEffect(() => {
    void load();
  }, [load]);

  const replaceRow = (updated: MatchRow) => {
    setRows((prev) => prev?.map((r) => (r.film_key === updated.film_key ? updated : r)) ?? null);
    // Refresh counts in the background; keep the edited row visible until then.
    api
      .matches(filter)
      .then((res) => setCounts(res.counts))
      .catch(() => {});
  };

  const needsReview = (counts.low_confidence ?? 0) + (counts.unmatched ?? 0) + (counts.error ?? 0);

  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <h1 className="text-2xl font-semibold">Match review</h1>
          <p className="text-zinc-400 text-sm mt-1">
            Fix films that didn't match TMDB, or matched with low confidence. Paste a TMDB id or a
            themoviedb.org URL.
          </p>
          <p className="text-zinc-500 text-xs mt-1">
            Fixes are saved in this browser and applied to your future uploads. To update your
            recommendations now, run processing again on the Upload page.
          </p>
        </div>
        <div className="flex gap-1 rounded-lg bg-zinc-900 p-1 text-sm">
          {(["review", "all"] as const).map((f) => (
            <button
              key={f}
              onClick={() => setFilter(f)}
              className={`px-3 py-1 rounded-md ${filter === f ? "bg-zinc-700 text-white" : "text-zinc-400 hover:text-white"}`}
            >
              {f === "review" ? `Needs review (${needsReview})` : "All films"}
            </button>
          ))}
        </div>
      </div>

      <div className="flex flex-wrap gap-2 text-xs">
        {Object.entries(counts).map(([status, n]) => (
          <span
            key={status}
            className={`rounded px-2 py-0.5 ${STATUS_STYLE[status as MatchStatus] ?? "bg-zinc-800 text-zinc-400"}`}
          >
            {STATUS_LABEL[status as MatchStatus] ?? status}: {n}
          </span>
        ))}
      </div>

      {error && (
        <div className="rounded-md bg-red-950/50 border border-red-800 p-3 text-red-300">{error}</div>
      )}

      {rows === null ? (
        <p className="text-zinc-500">Loading…</p>
      ) : rows.length === 0 ? (
        <p className="rounded-xl border border-zinc-800 p-8 text-center text-zinc-400">
          {filter === "review" ? "Nothing to review. Every film is matched." : "No films yet. Upload an export first."}
        </p>
      ) : (
        <ul className="divide-y divide-zinc-800 rounded-xl border border-zinc-800">
          {rows.map((row) => (
            <MatchItem key={row.film_key} row={row} onChange={replaceRow} />
          ))}
        </ul>
      )}
    </div>
  );
}

function MatchItem({ row, onChange }: { row: MatchRow; onChange: (r: MatchRow) => void }) {
  const [ref, setRef] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const act = async (fn: () => Promise<MatchRow>) => {
    setBusy(true);
    setError(null);
    try {
      onChange(await fn());
      setRef("");
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  const poster = posterUrl(row.movie?.poster_path ?? null);
  const status = row.status;

  return (
    <li className="flex gap-4 p-4">
      <div className="h-[69px] w-[46px] shrink-0 overflow-hidden rounded bg-zinc-800">
        {poster && <img src={poster} alt="" className="h-full w-full object-cover" loading="lazy" />}
      </div>

      <div className="min-w-0 flex-1 space-y-1">
        <div className="flex flex-wrap items-baseline gap-x-2">
          <span className="font-medium">{row.name}</span>
          <span className="text-zinc-500 text-sm">{row.year ?? "no year"}</span>
          {row.rating !== null && <span className="text-emerald-400 text-sm">{"★".repeat(Math.floor(row.rating))}{row.rating % 1 ? "½" : ""}</span>}
          {status && (
            <span className={`rounded px-1.5 py-0.5 text-xs ${STATUS_STYLE[status]}`}>
              {STATUS_LABEL[status]}
              {row.confidence !== null &&
                (status === "matched" || status === "low_confidence") &&
                ` · ${Math.round(row.confidence * 100)}%`}
            </span>
          )}
        </div>
        {row.movie ? (
          <div className="text-sm text-zinc-300">
            →{" "}
            <a
              href={`https://www.themoviedb.org/movie/${row.movie.tmdb_id}`}
              target="_blank"
              rel="noreferrer"
              className="underline decoration-zinc-600 hover:decoration-white"
            >
              {row.movie.title} ({row.movie.year ?? "?"})
            </a>
            {row.movie.directors.length > 0 && (
              <span className="text-zinc-500"> · {row.movie.directors.join(", ")}</span>
            )}
          </div>
        ) : (
          <div className="text-sm text-zinc-500">No TMDB film attached</div>
        )}
        {row.note && <div className="text-xs text-zinc-500">{row.note}</div>}
        {error && <div className="text-xs text-red-400">{error}</div>}
      </div>

      <div className="flex shrink-0 flex-col items-end gap-2">
        <form
          className="flex gap-2"
          onSubmit={(e) => {
            e.preventDefault();
            if (ref.trim()) void act(() => api.setMatch(row.film_key, ref.trim()));
          }}
        >
          <input
            value={ref}
            onChange={(e) => setRef(e.target.value)}
            placeholder="TMDB id or URL"
            className="w-40 rounded-md bg-zinc-900 border border-zinc-700 px-2 py-1 text-sm placeholder:text-zinc-600 focus:border-zinc-400 outline-none"
          />
          <button
            type="submit"
            disabled={busy || !ref.trim()}
            className="rounded-md bg-zinc-700 px-3 py-1 text-sm hover:bg-zinc-600 disabled:opacity-40"
          >
            Set
          </button>
        </form>
        <div className="flex gap-2">
          {row.tmdb_id !== null && status !== "manual" && status !== "matched" && (
            <button
              disabled={busy}
              onClick={() => void act(() => api.acceptMatch(row.film_key))}
              className="rounded-md bg-emerald-700 px-3 py-1 text-sm hover:bg-emerald-600 disabled:opacity-40"
            >
              Accept
            </button>
          )}
          {status !== "ignored" && (
            <button
              disabled={busy}
              onClick={() => void act(() => api.ignoreMatch(row.film_key))}
              className="rounded-md px-3 py-1 text-sm text-zinc-400 hover:text-white disabled:opacity-40"
              title="Not on TMDB / skip this film"
            >
              Ignore
            </button>
          )}
        </div>
      </div>
    </li>
  );
}
