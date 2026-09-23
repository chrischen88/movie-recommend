import type { Facets, RatingSource, RecFilters } from "../api";

const RATING_SOURCES: [RatingSource, string][] = [
  ["tmdb", "TMDB"],
  ["imdb", "IMDb"],
  ["rt", "Rotten Tomatoes"],
  ["metacritic", "Metacritic"],
];
const RATING_OPTIONS: Record<RatingSource, [number, string][]> = {
  tmdb: [6, 6.5, 7, 7.5, 8].map((r) => [r, `★ ${r.toFixed(1)}+`]),
  imdb: [6, 6.5, 7, 7.5, 8].map((r) => [r, `★ ${r.toFixed(1)}+`]),
  rt: [60, 75, 90].map((r) => [r, `${r}%+`]),
  metacritic: [60, 70, 80].map((r) => [r, `${r}+`]),
};
const RUNTIME_OPTIONS = [90, 120, 150];

const languageNames = (() => {
  try {
    return new Intl.DisplayNames(["en"], { type: "language" });
  } catch {
    return null;
  }
})();

const languageName = (code: string) => {
  try {
    return languageNames?.of(code) ?? code;
  } catch {
    return code;
  }
};

export function filtersFromParams(params: URLSearchParams): RecFilters {
  const num = (k: string) => {
    const v = params.get(k);
    return v !== null && v !== "" && !Number.isNaN(Number(v)) ? Number(v) : undefined;
  };
  const source = params.get("rating_source");
  return {
    min_rating: num("min_rating"),
    rating_source: RATING_SOURCES.some(([s]) => s === source) ? (source as RatingSource) : undefined,
    hide_low_quality: params.get("hide_low_quality") === "true" || undefined,
    genre: params.get("genre") || undefined,
    decade: num("decade"),
    max_runtime: num("max_runtime"),
    language: params.get("language") || undefined,
  };
}

export function filtersToParams(filters: RecFilters): URLSearchParams {
  const params = new URLSearchParams();
  for (const [k, v] of Object.entries(filters)) {
    if (v !== undefined && v !== "" && v !== false) params.set(k, String(v));
  }
  if (filters.min_rating === undefined) params.delete("rating_source"); // meaningless on its own
  return params;
}

export const hasFilters = (f: RecFilters) =>
  Object.entries(f).some(([k, v]) => k !== "rating_source" && v !== undefined && v !== "" && v !== false);

export default function FilterBar({
  filters,
  facets,
  onChange,
}: {
  filters: RecFilters;
  facets: Facets | undefined;
  onChange: (next: RecFilters) => void;
}) {
  const set = <K extends keyof RecFilters>(key: K, value: RecFilters[K]) =>
    onChange({ ...filters, [key]: value });
  const numOrUndef = (v: string) => (v === "" ? undefined : Number(v));
  const source: RatingSource = filters.rating_source ?? "tmdb";

  return (
    <div className="flex flex-wrap items-end gap-3 rounded-xl border border-zinc-800 bg-zinc-900/40 p-3">
      <Field label="Min rating">
        <div className="flex gap-1">
          <Select
            value={source}
            onChange={(v) => onChange({ ...filters, rating_source: v as RatingSource, min_rating: undefined })}
            options={RATING_SOURCES}
          />
          <Select
            value={filters.min_rating ?? ""}
            onChange={(v) => set("min_rating", numOrUndef(v))}
            options={[["", "Any"], ...RATING_OPTIONS[source].map(([r, label]) => [String(r), label] as [string, string])]}
          />
        </div>
      </Field>
      <Field label="Genre">
        <Select
          value={filters.genre ?? ""}
          onChange={(v) => set("genre", v || undefined)}
          options={[
            ["", "Any"],
            ...Object.entries(facets?.genres ?? {}).map(([g, n]) => [g, `${g} (${n})`] as [string, string]),
          ]}
        />
      </Field>
      <Field label="Decade">
        <Select
          value={filters.decade ?? ""}
          onChange={(v) => set("decade", numOrUndef(v))}
          options={[
            ["", "Any"],
            ...Object.entries(facets?.decades ?? {})
              .reverse()
              .map(([d, n]) => [d, `${d}s (${n})`] as [string, string]),
          ]}
        />
      </Field>
      <Field label="Max runtime">
        <Select
          value={filters.max_runtime ?? ""}
          onChange={(v) => set("max_runtime", numOrUndef(v))}
          options={[["", "Any"], ...RUNTIME_OPTIONS.map((m) => [String(m), `≤ ${m} min`] as [string, string])]}
        />
      </Field>
      <Field label="Language">
        <Select
          value={filters.language ?? ""}
          onChange={(v) => set("language", v || undefined)}
          options={[
            ["", "Any"],
            ...Object.entries(facets?.languages ?? {}).map(
              ([code, n]) => [code, `${languageName(code)} (${n})`] as [string, string],
            ),
          ]}
        />
      </Field>
      <label
        className="flex items-center gap-2 self-center pt-4 text-sm text-zinc-300"
        title="Hide films below Rotten Tomatoes 60% and IMDb 6.5 (TMDB 6.5 when neither is known)"
      >
        <input
          type="checkbox"
          checked={!!filters.hide_low_quality}
          onChange={(e) => set("hide_low_quality", e.target.checked || undefined)}
          className="accent-emerald-500"
        />
        Hide low quality
      </label>
      {hasFilters(filters) && (
        <button
          onClick={() => onChange({})}
          className="ml-auto rounded-md px-3 py-1.5 text-sm text-zinc-400 hover:text-white"
        >
          Clear filters
        </button>
      )}
    </div>
  );
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <label className="flex flex-col gap-1 text-xs text-zinc-500">
      {label}
      {children}
    </label>
  );
}

function Select({
  value,
  onChange,
  options,
}: {
  value: string | number;
  onChange: (v: string) => void;
  options: [string, string][];
}) {
  // Keep a value from the URL selectable even if the pool no longer has it.
  const all = options.some(([v]) => v === String(value)) ? options : [...options, [String(value), String(value)] as [string, string]];
  return (
    <select
      value={String(value)}
      onChange={(e) => onChange(e.target.value)}
      className="min-w-32 rounded-md border border-zinc-700 bg-zinc-900 px-2 py-1.5 text-sm text-zinc-200 outline-none focus:border-zinc-400"
    >
      {all.map(([v, label]) => (
        <option key={v} value={v}>
          {label}
        </option>
      ))}
    </select>
  );
}
