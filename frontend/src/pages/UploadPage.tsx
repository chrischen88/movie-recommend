import { useCallback, useEffect, useRef, useState, type DragEvent } from "react";
import { Link } from "react-router-dom";
import { api, type RunOut, type UploadResult } from "../api";

const POLL_MS = 800;

const STAGE_LABELS: Record<string, string> = {
  parsing: "Parsing export",
  matching: "Matching to TMDB",
  enrichment: "Fetching metadata",
  candidates: "Finding candidate films",
  embedding: "Embedding films",
  collab: "Training on MovieLens",
  taste: "Building your taste profile",
  blend: "Learning how to weigh the scores",
  omdb: "Fetching IMDb / Rotten Tomatoes ratings",
};

export default function UploadPage() {
  const [dragging, setDragging] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [upload, setUpload] = useState<UploadResult | null>(null);
  const [runOut, setRunOut] = useState<RunOut | null>(null);
  const timer = useRef<number | null>(null);

  const poll = useCallback((runId: number) => {
    if (timer.current) window.clearTimeout(timer.current);
    const tick = async () => {
      try {
        const out = await api.run(runId);
        setRunOut(out);
        if (out.run.status === "running") timer.current = window.setTimeout(tick, POLL_MS);
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
      }
    };
    void tick();
  }, []);

  useEffect(() => {
    api
      .latestRun()
      .then((out) => {
        if (!out) return;
        setRunOut(out);
        if (out.run.status === "running") poll(out.run.id);
      })
      .catch(() => {});
    return () => {
      if (timer.current) window.clearTimeout(timer.current);
    };
  }, [poll]);

  const doUpload = async (file: File) => {
    setUploading(true);
    setError(null);
    setUpload(null);
    try {
      const result = await api.upload(file);
      setUpload(result);
      poll(result.run_id);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setUploading(false);
    }
  };

  const resume = async () => {
    setError(null);
    try {
      const { run_id } = await api.resume();
      poll(run_id);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  };

  const onDrop = (e: DragEvent) => {
    e.preventDefault();
    setDragging(false);
    const file = e.dataTransfer.files[0];
    if (file) void doUpload(file);
  };

  const running = runOut?.run.status === "running";

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold">Upload your Letterboxd export</h1>
        <p className="text-zinc-400 text-sm mt-1">
          Letterboxd → Settings → Data → Export your data. Drop the ZIP here. Re-uploading a newer
          export only processes what changed.
        </p>
      </div>

      <label
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
        className={`flex flex-col items-center justify-center rounded-xl border-2 border-dashed p-12 transition ${
          running || uploading
            ? "pointer-events-none opacity-50 border-zinc-800"
            : dragging
              ? "cursor-pointer border-emerald-400 bg-emerald-950/30"
              : "cursor-pointer border-zinc-700 hover:border-zinc-500"
        }`}
      >
        <input
          type="file"
          accept=".zip"
          className="hidden"
          disabled={running || uploading}
          onChange={(e) => e.target.files?.[0] && void doUpload(e.target.files[0])}
        />
        <span className="text-zinc-300">
          {uploading ? "Uploading…" : running ? "Processing…" : "Drag & drop ZIP, or click to choose"}
        </span>
      </label>

      {error && (
        <div className="rounded-md bg-red-950/50 border border-red-800 p-3 text-red-300">{error}</div>
      )}

      {runOut && <RunPanel out={runOut} onResume={resume} />}

      {upload && upload.warnings.length > 0 && (
        <details className="text-sm rounded-xl border border-zinc-800 p-4">
          <summary className="cursor-pointer text-amber-400">
            {upload.warnings.length} parsing warning(s)
          </summary>
          <ul className="mt-2 space-y-1 text-zinc-400 list-disc pl-5">
            {upload.warnings.map((w, i) => (
              <li key={i}>{w}</li>
            ))}
          </ul>
        </details>
      )}
    </div>
  );
}

function RunPanel({ out, onResume }: { out: RunOut; onResume: () => void }) {
  const { run, stages } = out;
  const currentIdx = run.stage === "done" ? stages.length : stages.indexOf(run.stage);
  const needsReview =
    (run.stats.matching?.low_confidence ?? 0) +
    (run.stats.matching?.unmatched ?? 0) +
    (run.stats.matching?.error ?? 0);

  return (
    <div className="rounded-xl border border-zinc-800 p-5 space-y-5">
      <ol className="space-y-3">
        {stages.map((stage, i) => {
          const state =
            run.status === "error" && i === currentIdx
              ? "error"
              : i < currentIdx
                ? "done"
                : i === currentIdx && run.status === "running"
                  ? "active"
                  : "pending";
          const showProgress = state === "active" && run.progress_total > 0;
          return (
            <li key={stage} className="flex items-center gap-3">
              <span
                className={`h-6 w-6 shrink-0 rounded-full grid place-items-center text-xs font-semibold ${
                  state === "done"
                    ? "bg-emerald-600 text-white"
                    : state === "active"
                      ? "bg-sky-600 text-white animate-pulse"
                      : state === "error"
                        ? "bg-red-600 text-white"
                        : "bg-zinc-800 text-zinc-500"
                }`}
              >
                {state === "done" ? "✓" : state === "error" ? "!" : i + 1}
              </span>
              <div className="flex-1">
                <div className={state === "pending" ? "text-zinc-500" : ""}>
                  {STAGE_LABELS[stage] ?? stage}
                </div>
                {showProgress && (
                  <div className="mt-1.5 flex items-center gap-3">
                    <div className="h-1.5 flex-1 rounded bg-zinc-800 overflow-hidden">
                      <div
                        className="h-full bg-sky-500 transition-all"
                        style={{ width: `${(100 * run.progress_done) / run.progress_total}%` }}
                      />
                    </div>
                    <span className="text-xs text-zinc-400 tabular-nums">
                      {run.progress_done}/{run.progress_total}
                    </span>
                  </div>
                )}
              </div>
            </li>
          );
        })}
      </ol>

      {run.message && (
        <div
          className={`rounded-md p-3 text-sm ${
            run.status === "error"
              ? "bg-red-950/50 border border-red-800 text-red-300"
              : "bg-amber-950/40 border border-amber-800 text-amber-300"
          }`}
        >
          {run.message}
          {run.status !== "running" && (
            <button onClick={onResume} className="ml-3 underline hover:text-white">
              Run processing again
            </button>
          )}
        </div>
      )}

      {run.stats.reset && (
        <p className="rounded-md border border-sky-800 bg-sky-950/40 px-3 py-2 text-sm text-sky-200">
          This export is from a different account{run.stats.account ? ` (${run.stats.account})` : ""}, so it
          replaced all previous films, match fixes and recommendations instead of merging with them.
        </p>
      )}

      {run.status === "done" && (
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
          <Stat label="films" value={run.stats.films} />
          <Stat label="rated" value={run.stats.rated} />
          <Stat label="new / changed" value={`${run.stats.added ?? 0} / ${run.stats.changed ?? 0}`} />
          <Stat
            label="matched"
            value={
              run.stats.matching
                ? (run.stats.matching.matched ?? 0)
                : "—"
            }
          />
        </div>
      )}

      {run.status === "done" && (
        <Link
          to="/recommendations"
          className="mr-3 inline-block rounded-md bg-emerald-600 px-4 py-2 text-sm font-medium hover:bg-emerald-500"
        >
          See recommendations →
        </Link>
      )}

      {run.status === "done" && needsReview > 0 && (
        <Link
          to="/matches"
          className="inline-block rounded-md bg-amber-600 px-4 py-2 text-sm font-medium hover:bg-amber-500"
        >
          Review {needsReview} uncertain match{needsReview === 1 ? "" : "es"} →
        </Link>
      )}
    </div>
  );
}

function Stat({ label, value }: { label: string; value: number | string | undefined }) {
  return (
    <div className="rounded-lg bg-zinc-900 p-3">
      <div className="text-xs uppercase text-zinc-500">{label}</div>
      <div className="text-xl font-semibold tabular-nums">{value ?? "—"}</div>
    </div>
  );
}
