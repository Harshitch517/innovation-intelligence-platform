import { useState } from "react";
import { useSearchParams } from "react-router-dom";
import { api } from "../api";
import { DocCard, Empty, ErrorBox, Loading, Select, domainOptions } from "../components/ui";
import { useApi } from "../lib/useApi";

const EXAMPLES = [
  "deep learning for retinal disease detection",
  "robot gripper for soft objects",
  "intrusion detection in IoT networks",
  "perovskite solar cell stability",
  "large language models for question answering",
];
const YEARS = [2020, 2021, 2022, 2023, 2024, 2025];

export default function Search() {
  const [params, setParams] = useSearchParams();
  const q = params.get("q") || "";
  const filters = {
    domain: params.get("domain") || "",
    source: params.get("source") || "",
    year_from: params.get("year_from") || "",
    year_to: params.get("year_to") || "",
  };
  const [text, setText] = useState(q);

  const set = (patch) => {
    const next = new URLSearchParams(params);
    Object.entries(patch).forEach(([k, v]) => (v ? next.set(k, v) : next.delete(k)));
    setParams(next);
  };
  const { data, error, loading } = useApi(
    () => (q ? api.search({ q, k: 20, ...filters }) : Promise.resolve(null)),
    [q, filters.domain, filters.source, filters.year_from, filters.year_to],
  );

  return (
    <div className="space-y-5">
      <div>
        <h1 className="text-2xl font-semibold">Search innovations</h1>
        <p className="mt-1 text-sm text-ink-2">
          Natural-language search over 30,000 university patents and research papers, ranked by semantic similarity.
        </p>
      </div>
      <form
        onSubmit={(e) => {
          e.preventDefault();
          set({ q: text.trim() });
        }}
        className="flex gap-2"
      >
        <input
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="Describe a technology, e.g. wearable sensor for heart monitoring"
          className="min-w-0 flex-1 rounded-lg border border-line bg-surface px-3 py-2.5 text-ink placeholder:text-muted focus:outline-2 focus:outline-accent"
          aria-label="Search query"
        />
        <button className="btn-primary rounded-lg px-4 py-2.5 text-sm font-medium">Search</button>
      </form>

      <div className="flex flex-wrap items-end gap-3">
        <Select label="Domain" value={filters.domain} onChange={(v) => set({ domain: v })} options={domainOptions} />
        <Select
          label="Source"
          value={filters.source}
          onChange={(v) => set({ source: v })}
          options={[{ value: "", label: "Patents + papers" }, { value: "patent", label: "Patents" }, { value: "paper", label: "Papers" }]}
        />
        <Select label="From" value={filters.year_from} onChange={(v) => set({ year_from: v })}
          options={[{ value: "", label: "2020" }, ...YEARS.slice(1).map((y) => ({ value: String(y), label: String(y) }))]} />
        <Select label="To" value={filters.year_to} onChange={(v) => set({ year_to: v })}
          options={[{ value: "", label: "2025" }, ...YEARS.slice(0, -1).reverse().map((y) => ({ value: String(y), label: String(y) }))]} />
      </div>

      {!q && (
        <div className="flex flex-wrap gap-2">
          {EXAMPLES.map((e) => (
            <button key={e} onClick={() => { setText(e); set({ q: e }); }}
              className="rounded-full border border-line bg-surface px-3 py-1 text-sm text-ink-2 hover:border-accent hover:bg-surface-2 hover:text-ink">
              {e}
            </button>
          ))}
        </div>
      )}
      {error && <ErrorBox error={error} />}
      {q && loading && !data && <Loading label="Searching…" />}
      {data && (
        <div className={`space-y-3 transition-opacity ${loading ? "opacity-50" : ""}`}>
          <p className="text-xs text-muted">{data.count} results for “{data.query}”</p>
          {data.results.length === 0 && <Empty>No documents match these filters.</Empty>}
          {data.results.map((d) => <DocCard key={d.doc_id} doc={d} />)}
        </div>
      )}
    </div>
  );
}
