import { useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../api";
import { DOMAINS, fmt, useColors } from "../lib/theme";

export function DomainBadge({ domain }) {
  const c = useColors();
  return (
    <span className="inline-flex items-center gap-1.5 rounded-full border border-line px-2 py-0.5 text-xs font-medium text-ink-2">
      <span className="h-2 w-2 rounded-full" style={{ background: c.domain[domain] }} aria-hidden />
      {domain}
    </span>
  );
}

export function SourceBadge({ source }) {
  return (
    <span className="rounded bg-surface-2 px-1.5 py-0.5 text-[11px] font-semibold uppercase tracking-wide text-ink-2">
      {source}
    </span>
  );
}

export function Card({ title, subtitle, action, children, className = "" }) {
  return (
    <section className={`rounded-xl border border-line bg-surface p-4 sm:p-5 ${className}`}>
      {(title || action) && (
        <header className="mb-3 flex items-start justify-between gap-3">
          <div>
            {title && <h2 className="text-sm font-semibold text-ink">{title}</h2>}
            {subtitle && <p className="mt-0.5 text-xs text-muted">{subtitle}</p>}
          </div>
          {action}
        </header>
      )}
      {children}
    </section>
  );
}

// A chart with a "Table" toggle, so every value is reachable without hovering or colour.
export function ChartCard({ title, subtitle, columns, rows, children, className = "", dim = false }) {
  const [table, setTable] = useState(false);
  return (
    <Card
      title={title}
      subtitle={subtitle}
      className={className}
      action={
        rows?.length ? (
          <button
            onClick={() => setTable((t) => !t)}
            className="rounded-md border border-line px-2 py-1 text-xs text-ink-2 hover:bg-surface-2"
            aria-pressed={table}
          >
            {table ? "Chart" : "Table"}
          </button>
        ) : null
      }
    >
      {table ? (
        <div className="max-h-80 overflow-auto">
          <table className="tabular w-full text-left text-xs">
            <thead className="sticky top-0 bg-surface text-muted">
              <tr>{columns.map((c) => <th key={c} className="py-1 pr-3 font-medium">{c}</th>)}</tr>
            </thead>
            <tbody>
              {rows.map((r, i) => (
                <tr key={i} className="border-t border-line text-ink-2">
                  {r.map((v, j) => <td key={j} className="py-1 pr-3">{typeof v === "number" ? fmt(v) : v}</td>)}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <div className={`transition-opacity ${dim ? "opacity-50" : ""}`}>{children}</div>
      )}
    </Card>
  );
}

export function Loading({ label = "Loading…" }) {
  return (
    <div className="flex items-center gap-2 py-8 text-sm text-muted" role="status">
      <span className="h-4 w-4 animate-spin rounded-full border-2 border-line border-t-accent" />
      {label}
    </div>
  );
}

export function ErrorBox({ error }) {
  return (
    <div className="rounded-lg border border-line bg-surface p-4 text-sm" role="alert">
      <p className="font-medium text-critical">Something went wrong</p>
      <p className="mt-1 text-ink-2">{String(error?.message || error)}</p>
      <p className="mt-1 text-xs text-muted">Is the API running? (cd backend; uvicorn app.main:app)</p>
    </div>
  );
}

export function Empty({ children }) {
  return <p className="py-6 text-center text-sm text-muted">{children}</p>;
}

export function StatTile({ label, value, note }) {
  return (
    <div className="rounded-xl border border-line bg-surface p-4">
      <p className="text-xs font-medium text-muted">{label}</p>
      <p className="mt-1 text-2xl font-semibold text-ink">{fmt(value)}</p>
      {note && <p className="mt-0.5 text-xs text-muted">{note}</p>}
    </div>
  );
}

export function ScoreBar({ score }) {
  const pct = Math.max(0, Math.min(1, score)) * 100;
  return (
    <span className="inline-flex items-center gap-2 text-xs text-ink-2" title={`cosine similarity ${score.toFixed(3)}`}>
      <span className="h-1.5 w-16 overflow-hidden rounded-full bg-surface-2">
        <span className="block h-full rounded-full bg-accent" style={{ width: `${pct}%` }} />
      </span>
      <span className="tabular">{score.toFixed(2)}</span>
    </span>
  );
}

export function DocCard({ doc, children }) {
  return (
    <article className="rounded-lg border border-line bg-surface p-3 sm:p-4">
      <div className="flex flex-wrap items-center gap-2">
        <SourceBadge source={doc.source} />
        <DomainBadge domain={doc.domain} />
        <span className="tabular text-xs text-muted">{doc.year}</span>
        {doc.score !== undefined && <span className="ml-auto"><ScoreBar score={doc.score} /></span>}
      </div>
      <Link to={`/documents/${encodeURIComponent(doc.doc_id)}`} className="mt-2 block font-medium leading-snug text-ink hover:text-accent">
        {doc.title}
      </Link>
      <p className="mt-1 text-sm text-ink-2">
        <Link to={`/institutions?ids=${encodeURIComponent(doc.institution)}`} className="hover:underline">
          {doc.institution_name}
        </Link>
        {doc.country && <span className="text-muted"> · {doc.country}</span>}
        {doc.other_institution && <span className="ml-2 whitespace-nowrap rounded bg-surface-2 px-1.5 py-0.5 text-[11px] text-ink-2">other institution</span>}
      </p>
      {children}
    </article>
  );
}

// "Explain" button: asks the API (Gemini, cached) why two documents are similar.
export function Explain({ a, b }) {
  const [state, setState] = useState({ status: "idle" });
  const run = () => {
    setState({ status: "loading" });
    api.explain(a, b).then((r) => setState({ status: "done", ...r })).catch((e) => setState({ status: "error", error: e }));
  };
  if (state.status === "idle")
    return <button onClick={run} className="text-xs font-medium text-accent hover:underline">Explain similarity</button>;
  if (state.status === "loading") return <span className="text-xs text-muted">Asking Gemini…</span>;
  if (state.status === "error") return <span className="text-xs text-critical">{state.error.message}</span>;
  return (
    <div className="mt-2 rounded-md bg-surface-2 p-3 text-sm text-ink-2">
      {state.explanation ?? <span className="text-muted">{state.note}</span>}
      {state.explanation && <p className="mt-1 text-[11px] text-muted">{state.cached ? "cached" : "generated"} · {state.model}</p>}
    </div>
  );
}

export function Select({ label, value, onChange, options, className = "" }) {
  return (
    <label className={`flex flex-col gap-1 text-xs text-muted ${className}`}>
      {label}
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="rounded-md border border-line bg-surface px-2 py-1.5 text-sm text-ink"
      >
        {options.map((o) => (
          <option key={o.value} value={o.value}>{o.label}</option>
        ))}
      </select>
    </label>
  );
}

export const domainOptions = [{ value: "", label: "All domains" }, ...DOMAINS.map((d) => ({ value: d.name, label: d.name }))];

// Recharts tooltip: value first (strong), series name second, keyed by a short line.
export function ChartTooltip({ active, payload, label, labelFormatter }) {
  if (!active || !payload?.length) return null;
  return (
    <div className="rounded-md border border-line bg-surface px-3 py-2 text-xs shadow-sm">
      <p className="mb-1 text-muted">{labelFormatter ? labelFormatter(label) : label}</p>
      {payload.map((p) => (
        <p key={p.dataKey ?? p.name} className="flex items-center gap-2">
          <span className="h-0.5 w-3 rounded" style={{ background: p.color || p.fill }} aria-hidden />
          <span className="tabular font-semibold text-ink">{fmt(p.value)}</span>
          <span className="text-ink-2">{p.name}</span>
        </p>
      ))}
    </div>
  );
}
