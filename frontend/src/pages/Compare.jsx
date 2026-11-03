import { useRef, useState } from "react";
import { Link, useSearchParams } from "react-router-dom";
import { api } from "../api";
import { Card, DomainBadge, ErrorBox, Explain, Loading, SourceBadge } from "../components/ui";
import { useApi } from "../lib/useApi";

// Cosine similarity shown as a matching percentage, with a plain-language band.
const matchPct = (s) => Math.max(0, Math.min(100, Math.round(s * 100)));
function band(s) {
  const p = matchPct(s);
  if (p >= 75) return "Very high match";
  if (p >= 60) return "High match";
  if (p >= 45) return "Moderate match";
  return "Low match";
}

function MatchMeter({ score, size = "sm" }) {
  const p = matchPct(score);
  return (
    <span className={`inline-flex items-center gap-2 ${size === "lg" ? "text-sm" : "text-xs"}`} title={`cosine similarity ${score.toFixed(3)}`}>
      <span className={`${size === "lg" ? "h-2 w-40" : "h-1.5 w-14"} overflow-hidden rounded-full bg-surface-2`}>
        <span className="block h-full rounded-full bg-accent" style={{ width: `${p}%` }} />
      </span>
      <span className="tabular font-semibold text-ink">{p}%</span>
    </span>
  );
}

// Preview of a hovered or focused result: badges, institution and the start of the abstract.
function Preview({ doc, hint }) {
  if (!doc)
    return <p className="rounded-lg border border-dashed border-line p-3 text-xs text-muted">{hint}</p>;
  return (
    <div className="rounded-lg border border-line bg-surface-2 p-3" aria-live="polite">
      <div className="flex flex-wrap items-center gap-2">
        <SourceBadge source={doc.source} />
        <DomainBadge domain={doc.domain} />
        <span className="tabular text-xs text-muted">{doc.year}</span>
        {doc.score !== undefined && <span className="ml-auto"><MatchMeter score={doc.score} /></span>}
      </div>
      <p className="mt-2 text-sm font-medium leading-snug text-ink">{doc.title}</p>
      <p className="mt-0.5 text-xs text-ink-2">{doc.institution_name}{doc.country ? ` · ${doc.country}` : ""}</p>
      {doc.snippet && <p className="mt-2 line-clamp-4 text-xs leading-relaxed text-ink-2">{doc.snippet}…</p>}
    </div>
  );
}

// Clickable result rows; hovering or focusing a row shows its preview below the list.
function ResultList({ items, onPick, selectedId, hint = "Hover over a result to preview it." }) {
  const [hovered, setHovered] = useState(null);
  return (
    <div className="mt-2 space-y-2">
      <ul className="max-h-72 divide-y divide-line overflow-auto rounded-lg border border-line" onMouseLeave={() => setHovered(null)}>
        {items.map((r) => (
          <li key={r.doc_id}>
            <button
              onClick={() => onPick(r.doc_id)}
              onMouseEnter={() => setHovered(r)}
              onFocus={() => setHovered(r)}
              className={`row-btn flex w-full items-center gap-3 px-3 py-2 text-left text-sm ${selectedId === r.doc_id ? "bg-surface-2" : ""}`}
            >
              <span className="min-w-0 flex-1">
                <span className="line-clamp-2 text-ink">{r.title}</span>
                <span className="text-xs text-muted">{r.institution_name} · {r.year}</span>
              </span>
              {r.score !== undefined && <MatchMeter score={r.score} />}
            </button>
          </li>
        ))}
      </ul>
      <Preview doc={hovered} hint={hint} />
    </div>
  );
}

// The chosen database document, with a fuller preview and a way to change it.
function SelectedDoc({ id, onClear }) {
  const { data, error } = useApi(() => api.document(id), [id]);
  if (error) return <ErrorBox error={error} />;
  if (!data) return <Loading />;
  return (
    <div className="space-y-2">
      <div className="flex flex-wrap items-center gap-2">
        <SourceBadge source={data.source} />
        <DomainBadge domain={data.domain} />
        <span className="tabular text-xs text-muted">{data.year}</span>
        <button onClick={onClear} className="btn-link ml-auto rounded px-1 text-xs font-medium text-accent">Change</button>
      </div>
      <Link to={`/documents/${encodeURIComponent(data.doc_id)}`} className="block font-medium leading-snug text-ink hover:text-accent">
        {data.title}
      </Link>
      <p className="text-xs text-ink-2">{data.institution_name}{data.country ? ` · ${data.country}` : ""}</p>
      <p className="line-clamp-5 text-xs leading-relaxed text-ink-2">{data.abstract}</p>
    </div>
  );
}

function SearchPicker({ value, onPick }) {
  const [text, setText] = useState("");
  const [results, setResults] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const run = (e) => {
    e.preventDefault();
    if (text.trim().length < 2) return;
    setBusy(true);
    setError(null);
    api.search({ q: text.trim(), k: 10 }).then((r) => setResults(r.results)).catch(setError).finally(() => setBusy(false));
  };
  if (value) return <SelectedDoc id={value} onClear={() => onPick("")} />;
  return (
    <div>
      <form onSubmit={run} className="flex gap-2">
        <input value={text} onChange={(e) => setText(e.target.value)} placeholder="Search patents and papers…" aria-label="Search documents"
          className="min-w-0 flex-1 rounded-md border border-line bg-surface px-2.5 py-1.5 text-sm" />
        <button className="btn-primary rounded-md px-3 text-sm font-medium" disabled={busy}>{busy ? "…" : "Find"}</button>
      </form>
      {error && <p className="mt-2 text-xs text-critical">{error.message}</p>}
      {results && results.length === 0 && <p className="mt-2 text-xs text-muted">No results. Try other words, or upload your document.</p>}
      {results && results.length > 0 && (
        <ResultList items={results} onPick={onPick} hint="Hover over a result to preview it; click to select. The bar shows how well it matches your search." />
      )}
    </div>
  );
}

// Upload a Word/PDF/text file. Text is extracted on the server without any AI call.
function UploadPicker({ upload, setUpload, onPickMatch, selectedB }) {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const [drag, setDrag] = useState(false);
  const input = useRef(null);
  const send = (file) => {
    if (!file) return;
    setBusy(true);
    setError(null);
    api.analyze(file, 10).then(setUpload).catch(setError).finally(() => setBusy(false));
  };
  if (upload)
    return (
      <div className="space-y-3">
        <div className="flex flex-wrap items-center gap-2">
          <span className="rounded bg-surface-2 px-1.5 py-0.5 text-[11px] font-semibold uppercase tracking-wide text-ink-2">your {upload.file_type}</span>
          <DomainBadge domain={upload.prediction.domain} />
          <span className="text-xs text-muted">predicted domain · {upload.word_count.toLocaleString()} words</span>
          <button onClick={() => setUpload(null)} className="btn-link ml-auto rounded px-1 text-xs font-medium text-accent">Change file</button>
        </div>
        <p className="font-medium leading-snug text-ink">{upload.title}</p>
        <details className="text-xs text-ink-2">
          <summary className="cursor-pointer text-muted hover:text-ink">
            {upload.method === "abstract_section" ? "Abstract found in your document" : "Opening text used (no abstract heading found)"}
          </summary>
          <p className="mt-1 leading-relaxed">{upload.abstract}</p>
        </details>
        <div>
          <p className="text-xs font-medium text-muted">Closest documents in the database: pick one as document B</p>
          <ResultList items={upload.matches} onPick={onPickMatch} selectedId={selectedB}
            hint="Hover over a match to preview it; the bar shows the matching percentage with your document." />
        </div>
      </div>
    );
  return (
    <div>
      <div
        onDragOver={(e) => { e.preventDefault(); setDrag(true); }}
        onDragLeave={() => setDrag(false)}
        onDrop={(e) => { e.preventDefault(); setDrag(false); send(e.dataTransfer.files[0]); }}
        onClick={() => input.current?.click()}
        onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && input.current?.click()}
        role="button"
        tabIndex={0}
        className={`flex flex-col items-center gap-1 rounded-lg border-2 border-dashed px-4 py-8 text-center hover:border-accent hover:bg-surface-2 ${drag ? "border-accent bg-surface-2" : "border-line"}`}
      >
        {busy ? <Loading label="Reading your document…" /> : (
          <>
            <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" className="text-muted" aria-hidden>
              <path d="M12 16V4m0 0L7 9m5-5 5 5M4 16v3a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-3" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
            <p className="text-sm font-medium text-ink">Drop a Word or PDF file here, or click to choose</p>
            <p className="text-xs text-muted">.docx, .pdf or .txt · up to 5 MB · title and abstract are read automatically</p>
          </>
        )}
      </div>
      <input ref={input} type="file" accept=".pdf,.docx,.txt" className="hidden" onChange={(e) => send(e.target.files[0])} />
      {error && <p className="mt-2 text-sm text-critical">{error.message.replace(/^\d+: /, "")}</p>}
    </div>
  );
}

function Tabs({ value, onChange }) {
  const tab = (key, label) => (
    <button
      onClick={() => onChange(key)}
      aria-pressed={value === key}
      className={`rounded-md px-2.5 py-1 text-xs font-medium ${value === key ? "bg-surface-2 text-ink" : "text-ink-2 hover:bg-surface-2 hover:text-ink"}`}
    >
      {label}
    </button>
  );
  return <div className="flex gap-1">{tab("db", "Search database")}{tab("upload", "Upload file")}</div>;
}

const Flag = ({ ok, children }) => (
  <li className="flex items-center gap-2 text-sm">
    <span className={`font-semibold ${ok ? "text-good" : "text-muted"}`} aria-hidden>{ok ? "✓" : "–"}</span>
    <span className={ok ? "text-ink" : "text-muted"}>{children}</span>
  </li>
);

function ScoreBlock({ similarity }) {
  return (
    <div className="text-center">
      <p className="text-xs text-muted">matching</p>
      <p className="tabular text-5xl font-semibold text-ink">{matchPct(similarity)}%</p>
      <p className="mt-1 text-sm font-medium text-ink-2">{band(similarity)}</p>
      <div className="mt-2 flex justify-center"><MatchMeter score={similarity} size="lg" /></div>
    </div>
  );
}

function DbResult({ a, b }) {
  const { data, error, loading } = useApi(() => api.compare(a, b), [a, b]);
  if (error) return <ErrorBox error={error} />;
  if (!data) return <Loading />;
  return (
    <Card className={loading ? "opacity-50" : ""}>
      <ScoreBlock similarity={data.similarity} />
      <div className="mt-5 grid gap-4 border-t border-line pt-4 md:grid-cols-2">
        <ul className="space-y-1">
          <Flag ok={data.same_domain}>Same domain</Flag>
          <Flag ok={data.same_cluster}>Same sub-area</Flag>
          <Flag ok={data.same_global_cluster}>Same global cluster</Flag>
          <Flag ok={data.different_institutions}>Different institutions</Flag>
        </ul>
        <div>
          <p className="text-xs text-muted">Shared topic keywords</p>
          <p className="mt-1 text-sm text-ink-2">{data.shared_keywords.length ? data.shared_keywords.join(", ") : "none"}</p>
        </div>
      </div>
      <div className="mt-4 border-t border-line pt-4"><Explain key={`${a}|${b}`} a={a} b={b} /></div>
    </Card>
  );
}

function UploadResult({ upload, b }) {
  const { data, error, loading } = useApi(() => api.compareUpload(upload.title, upload.abstract, b), [upload, b]);
  if (error) return <ErrorBox error={error} />;
  if (!data) return <Loading />;
  return (
    <Card className={loading ? "opacity-50" : ""}>
      <ScoreBlock similarity={data.similarity} />
      <ul className="mt-5 space-y-1 border-t border-line pt-4">
        <Flag ok={data.same_domain}>
          Same domain (your document: {data.predicted_domain}; database document: {data.b.domain})
        </Flag>
      </ul>
      <div className="mt-4 border-t border-line pt-4">
        <Explain key={`${upload.title}|${b}`} load={() => api.explainUpload(upload.title, upload.abstract, b)} />
      </div>
    </Card>
  );
}

export default function Compare() {
  const [params, setParams] = useSearchParams();
  const a = params.get("a") || "";
  const b = params.get("b") || "";
  const [modeA, setModeA] = useState("db");
  const [upload, setUpload] = useState(null);
  const set = (key) => (id) => {
    const n = new URLSearchParams(params);
    id ? n.set(key, id) : n.delete(key);
    setParams(n);
  };
  const usingUpload = modeA === "upload";

  return (
    <div className="space-y-5">
      <div>
        <h1 className="text-2xl font-semibold">Compare two innovations</h1>
        <p className="mt-1 text-sm text-ink-2">
          Pick two documents from the database, or upload your own patent or paper and compare it with the closest matches.
        </p>
      </div>
      <div className="grid gap-4 md:grid-cols-2">
        <Card title="Document A" action={<Tabs value={modeA} onChange={setModeA} />}>
          {usingUpload ? (
            <UploadPicker upload={upload} setUpload={setUpload} onPickMatch={set("b")} selectedB={b} />
          ) : (
            <SearchPicker value={a} onPick={set("a")} />
          )}
        </Card>
        <Card title="Document B" subtitle={usingUpload && upload && !b ? "Select one of the matches on the left, or search" : undefined}>
          <SearchPicker value={b} onPick={set("b")} />
        </Card>
      </div>
      {usingUpload && upload && b && <UploadResult upload={upload} b={b} />}
      {!usingUpload && a && b && <DbResult a={a} b={b} />}
    </div>
  );
}
