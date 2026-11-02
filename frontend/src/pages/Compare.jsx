import { useState } from "react";
import { Link, useSearchParams } from "react-router-dom";
import { api } from "../api";
import { Card, DomainBadge, ErrorBox, Explain, Loading, SourceBadge } from "../components/ui";
import { useApi } from "../lib/useApi";

// Pick a document by searching for it.
function Picker({ label, value, onPick }) {
  const [text, setText] = useState("");
  const [results, setResults] = useState(null);
  const [busy, setBusy] = useState(false);
  const run = (e) => {
    e.preventDefault();
    if (text.trim().length < 2) return;
    setBusy(true);
    api.search({ q: text.trim(), k: 6 }).then((r) => setResults(r.results)).finally(() => setBusy(false));
  };
  return (
    <Card title={label} subtitle={value ? `Selected: ${value}` : "Search for a document to compare"}>
      <form onSubmit={run} className="flex gap-2">
        <input value={text} onChange={(e) => setText(e.target.value)} placeholder="Search…" aria-label={`${label} search`}
          className="min-w-0 flex-1 rounded-md border border-line bg-surface px-2.5 py-1.5 text-sm" />
        <button className="rounded-md border border-line px-3 text-sm hover:bg-surface-2">{busy ? "…" : "Find"}</button>
      </form>
      {results && (
        <ul className="mt-2 divide-y divide-line">
          {results.map((r) => (
            <li key={r.doc_id}>
              <button onClick={() => { onPick(r.doc_id); setResults(null); }} className="w-full py-2 text-left text-sm hover:text-accent">
                {r.title} <span className="text-xs text-muted">· {r.institution_name}, {r.year}</span>
              </button>
            </li>
          ))}
        </ul>
      )}
    </Card>
  );
}

function Side({ doc }) {
  return (
    <div className="space-y-1">
      <div className="flex flex-wrap items-center gap-2"><SourceBadge source={doc.source} /><DomainBadge domain={doc.domain} /><span className="tabular text-xs text-muted">{doc.year}</span></div>
      <Link to={`/documents/${encodeURIComponent(doc.doc_id)}`} className="block font-medium text-ink hover:text-accent">{doc.title}</Link>
      <p className="text-sm text-ink-2">{doc.institution_name}</p>
    </div>
  );
}

const Flag = ({ ok, children }) => (
  <li className="flex items-center gap-2 text-sm">
    <span className={`font-semibold ${ok ? "text-good" : "text-muted"}`} aria-hidden>{ok ? "✓" : "–"}</span>
    <span className={ok ? "text-ink" : "text-muted"}>{children}</span>
  </li>
);

export default function Compare() {
  const [params, setParams] = useSearchParams();
  const a = params.get("a") || "";
  const b = params.get("b") || "";
  const pick = (key) => (id) => { const n = new URLSearchParams(params); n.set(key, id); setParams(n); };
  const cmp = useApi(() => (a && b ? api.compare(a, b) : Promise.resolve(null)), [a, b]);

  return (
    <div className="space-y-5">
      <div>
        <h1 className="text-2xl font-semibold">Compare two innovations</h1>
        <p className="mt-1 text-sm text-ink-2">Cosine similarity of their embeddings, what they share, and an AI explanation.</p>
      </div>
      <div className="grid gap-4 md:grid-cols-2">
        <Picker label="Document A" value={a} onPick={pick("a")} />
        <Picker label="Document B" value={b} onPick={pick("b")} />
      </div>
      {cmp.error && <ErrorBox error={cmp.error} />}
      {a && b && !cmp.data && !cmp.error && <Loading />}
      {cmp.data && (
        <Card className={cmp.loading ? "opacity-50" : ""}>
          <div className="grid gap-5 md:grid-cols-[1fr_auto_1fr] md:items-center">
            <Side doc={cmp.data.a} />
            <div className="text-center">
              <p className="text-xs text-muted">similarity</p>
              <p className="tabular text-4xl font-semibold">{cmp.data.similarity.toFixed(2)}</p>
              <div className="mx-auto mt-2 h-1.5 w-32 overflow-hidden rounded-full bg-surface-2">
                <div className="h-full rounded-full bg-accent" style={{ width: `${Math.max(0, cmp.data.similarity) * 100}%` }} />
              </div>
            </div>
            <Side doc={cmp.data.b} />
          </div>
          <div className="mt-5 grid gap-4 border-t border-line pt-4 md:grid-cols-2">
            <ul className="space-y-1">
              <Flag ok={cmp.data.same_domain}>Same domain</Flag>
              <Flag ok={cmp.data.same_cluster}>Same sub-area</Flag>
              <Flag ok={cmp.data.same_global_cluster}>Same global cluster</Flag>
              <Flag ok={cmp.data.different_institutions}>Different institutions</Flag>
            </ul>
            <div>
              <p className="text-xs text-muted">Shared topic keywords</p>
              <p className="mt-1 text-sm text-ink-2">{cmp.data.shared_keywords.length ? cmp.data.shared_keywords.join(", ") : "none"}</p>
            </div>
          </div>
          <div className="mt-4 border-t border-line pt-4">
            <Explain key={`${a}|${b}`} a={a} b={b} />
          </div>
        </Card>
      )}
    </div>
  );
}
