import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api, type LinearBlend, type MetricsResponse } from "../api";

const METHODS: [string, string][] = [
  ["baseline_mean", "Your average rating (baseline)"],
  ["profile", "① Taste profile alone"],
  ["embedding", "② Embedding similarity alone"],
  ["collab", "③ Collaborative alone"],
  ["fixed_blend", "Fixed weights"],
  ["learned_blend", "Learned blend"],
];

const FEATURES: Record<string, string> = {
  profile: "① Taste profile",
  embedding: "② Similarity",
  collab: "③ Collaborative",
  votes: "Popularity (log votes)",
};

const SOURCES: Record<string, string> = {
  recommendations: "TMDB recommendations for films you rated highly",
  similar: "TMDB similar films for films you rated highly",
  discover: "Top films in your genres and languages",
  collab: "Top predictions from MovieLens users like you",
};

export default function MetricsPage() {
  const [data, setData] = useState<MetricsResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api
      .metrics()
      .then(setData)
      .catch((e) => setError(e instanceof Error ? e.message : String(e)));
  }, []);

  if (error) {
    return <div className="rounded-md bg-red-950/50 border border-red-800 p-3 text-red-300">{error}</div>;
  }
  if (!data) return <p className="text-zinc-500">Loading…</p>;
  if (!data.ready) {
    return (
      <div className="rounded-xl border border-zinc-800 p-8 text-center space-y-3">
        <p className="text-zinc-300">{data.message ?? "No metrics yet."}</p>
        <Link to="/" className="inline-block rounded-md bg-emerald-600 px-4 py-2 text-sm hover:bg-emerald-500">
          Upload your export
        </Link>
      </div>
    );
  }

  const metrics = data.metrics ?? {};
  const bestRmse = Math.min(...Object.values(metrics).map((m) => m.rmse));

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-2xl font-semibold">How well it works</h1>
        <p className="text-zinc-400 text-sm mt-1">
          Each score is tested on your own ratings with 5-fold cross-fitting: your rated films are split into five
          groups, and each group is scored by a model built from the other four, so no film helps predict itself.
          {data.trained_at && ` Last trained ${new Date(data.trained_at).toLocaleString()}.`}
        </p>
      </div>

      <Section title="Blend">
        <p className="text-sm text-zinc-300">
          {data.mode === "learned" ? (
            <>
              Using the <b>learned blend</b>: weights fitted to your {data.n_ratings} rated films
              {data.n_with_collab ? ` (${data.n_with_collab} of them are in MovieLens)` : ""}.
            </>
          ) : (
            <>
              Using <b>fixed weights</b>{" "}
              {Object.entries(data.fixed_weights ?? {})
                .map(([k, w]) => `${FEATURES[k] ?? k} ${w}`)
                .join(", ")}
              . The blend is learned once you've rated at least {data.min_ratings_for_learning} films (you have{" "}
              {data.n_ratings}).
            </>
          )}
        </p>
        {data.mode === "learned" && (
          <div className="grid gap-4 sm:grid-cols-2">
            {data.full && <Weights title="Films in MovieLens" model={data.full} />}
            {data.partial && <Weights title="Films not in MovieLens" model={data.partial} />}
          </div>
        )}
      </Section>

      {Object.keys(metrics).length > 0 && (
        <Section title="Prediction accuracy on your held-out ratings">
          <table className="w-full text-sm">
            <thead className="text-left text-xs text-zinc-500">
              <tr>
                <th className="py-1 font-normal">Method</th>
                <th className="py-1 font-normal text-right" title="Root-mean-square error in stars; lower is better">
                  RMSE (★)
                </th>
                <th
                  className="py-1 font-normal text-right"
                  title="Spearman rank correlation with your ratings; higher is better. This is what matters for ranking."
                >
                  Rank correlation
                </th>
                <th className="py-1 font-normal text-right">Films</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-zinc-800">
              {METHODS.filter(([k]) => metrics[k]).map(([key, label]) => {
                const m = metrics[key];
                return (
                  <tr key={key} className={key === "learned_blend" ? "text-emerald-300" : "text-zinc-300"}>
                    <td className="py-1.5">{label}</td>
                    <td className={`py-1.5 text-right tabular-nums ${m.rmse === bestRmse ? "font-semibold" : ""}`}>
                      {m.rmse.toFixed(3)}
                    </td>
                    <td className="py-1.5 text-right tabular-nums">
                      {m.spearman === null ? "—" : m.spearman.toFixed(3)}
                    </td>
                    <td className="py-1.5 text-right tabular-nums text-zinc-500">{m.n}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
          <p className="text-xs text-zinc-500">
            RMSE is in stars (lower is better). Rank correlation runs from −1 to 1 (higher is better). Single scores
            are mapped to stars with a linear fit, so their RMSE is comparable with the blends. ③ is measured only on
            films that are in MovieLens.
          </p>
        </Section>
      )}

      <div className="grid gap-8 sm:grid-cols-2">
        <Section title="Candidates">
          <p className="text-sm text-zinc-300">{data.candidates?.total.toLocaleString()} unseen films are scored.</p>
          <ul className="space-y-1 text-sm">
            {Object.entries(data.candidates?.by_source ?? {}).map(([kind, n]) => (
              <li key={kind} className="flex justify-between gap-2">
                <span className="text-zinc-400">{SOURCES[kind] ?? kind}</span>
                <span className="tabular-nums">{n.toLocaleString()}</span>
              </li>
            ))}
          </ul>
          <p className="text-xs text-zinc-500">A film can come from several sources. Your own watchlist is always included.</p>
        </Section>

        <Section title="Data sources">
          <dl className="space-y-1 text-sm">
            {data.collab_model ? (
              <>
                <Row label="MovieLens dataset" value={data.collab_model.dataset} />
                <Row
                  label="MovieLens model RMSE"
                  value={`${data.collab_model.val_rmse.toFixed(3)} (average-rating baseline ${data.collab_model.val_rmse_global_mean.toFixed(3)})`}
                />
                <Row
                  label="MovieLens size"
                  value={`${data.collab_model.n_users.toLocaleString()} users · ${data.collab_model.n_items.toLocaleString()} films`}
                />
              </>
            ) : (
              <Row label="MovieLens" value="not trained (make train)" />
            )}
            <Row
              label="OMDb requests today"
              value={
                data.omdb?.enabled
                  ? `${data.omdb.used_today.toLocaleString()} of ${data.omdb.daily_limit.toLocaleString()}`
                  : "disabled (no OMDB_API_KEY)"
              }
            />
          </dl>
        </Section>
      </div>
    </div>
  );
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section className="space-y-3">
      <h2 className="text-sm font-medium uppercase tracking-wide text-zinc-500">{title}</h2>
      {children}
    </section>
  );
}

function Row({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex justify-between gap-4">
      <dt className="text-zinc-400">{label}</dt>
      <dd className="text-right">{value}</dd>
    </div>
  );
}

function Weights({ title, model }: { title: string; model: LinearBlend }) {
  const max = Math.max(...model.coef, 1e-9);
  return (
    <div className="rounded-xl border border-zinc-800 p-4 space-y-2">
      <h3 className="text-sm text-zinc-300">{title}</h3>
      {model.features.map((f, i) => (
        <div key={f} title={f === "votes" ? `${model.coef[i].toFixed(3)}★ per standard deviation of log vote count` : `${model.coef[i].toFixed(3)}★ between the bottom and top of this score`}>
          <div className="flex justify-between text-xs text-zinc-500">
            <span>{FEATURES[f] ?? f}</span>
            <span className="tabular-nums">{model.coef[i].toFixed(2)}</span>
          </div>
          <div className="mt-0.5 h-1 rounded bg-zinc-800">
            <div className="h-full rounded bg-emerald-500" style={{ width: `${(model.coef[i] / max) * 100}%` }} />
          </div>
        </div>
      ))}
      <p className="text-[11px] text-zinc-600">
        Stars gained from the bottom to the top of each score (per standard deviation for popularity). Weights can't be negative. Ridge α = {model.alpha}.
      </p>
    </div>
  );
}
