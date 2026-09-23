import type { Facets, RecFilters } from "../api";

const RATING_OPTIONS = [6, 6.5, 7, 7.5, 8];
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
  return {
    min_rating: num("min_rating"),
    genre: params.get("genre") || undefined,
    decade: num("decade"),
    max_runtime: num("max_runtime"),
    language: params.get("language") || undefined,
  };
}

export function filtersToParams(filters: RecFilters): URLSearchParams {
  const params = new URLSearchParams();
  for (const [k, v] of Object.entries(filters)) {
    if (v !== undefined && v !== "") params.set(k, String(v));
  }
  return params;
}

export const hasFilters = (f: RecFilters) => Object.values(f).some((v) => v !== undefined && v !== "");

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

  return (
    <div className="flex flex-wrap items-end gap-3 rounded-xl border border-zinc-800 bg-zinc-900/40 p-3">
      <Field label="Min rating (TMDB)">
        <Select
          value={filters.min_rating ?? ""}
          onChange={(v) => set("min_rating", numOrUndef(v))}
          options={[["", "Any"], ...RATING_OPTIONS.map((r) => [String(r), `★ ${r.toFixed(1)}+`] as [string, string])]}
        />
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
