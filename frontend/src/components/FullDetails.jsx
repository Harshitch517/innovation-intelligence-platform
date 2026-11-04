import { useEffect, useRef, useState } from "react";
import { api } from "../api";
import { detailsCache, downloadPdf, pdfCache, PDF_CACHE_LIMIT } from "../lib/docCache";
import { fmt } from "../lib/theme";
import { Card } from "./ui";

const mb = (n) => `${(n / (1024 * 1024)).toFixed(1)} MB`;

function Fact({ label, children }) {
  if (children === null || children === undefined || children === "") return null;
  return (
    <div>
      <dt className="text-xs text-muted">{label}</dt>
      <dd className="text-sm text-ink">{children}</dd>
    </div>
  );
}

function Facts({ d, codes }) {
  const more = d.author_count - d.authors.length;
  return (
    <dl className="grid gap-3 sm:grid-cols-2">
      {d.authors.length > 0 && (
        <div className="sm:col-span-2">
          <dt className="text-xs text-muted">Authors</dt>
          <dd className="text-sm text-ink">{d.authors.join(", ")}{more > 0 ? ` and ${more} more` : ""}</dd>
        </div>
      )}
      <Fact label="Published in">{d.venue}</Fact>
      <Fact label="Publication date">{d.publication_date}</Fact>
      <Fact label="Cited by">{d.cited_by_count != null ? `${fmt(d.cited_by_count)} works` : null}</Fact>
      <Fact label="References">{d.references_count != null ? fmt(d.references_count) : null}</Fact>
      <Fact label="Access">{d.kind === "paper" ? (d.is_open_access ? `Open access${d.oa_status ? ` (${d.oa_status})` : ""}` : "Subscription") : null}</Fact>
      <Fact label="DOI">{d.doi && <a href={d.doi} target="_blank" rel="noreferrer" className="break-all text-accent hover:underline">{d.doi.replace("https://doi.org/", "")}</a>}</Fact>
      {d.kind === "patent" && (
        <div className="sm:col-span-2">
          <dt className="text-xs text-muted">CPC codes</dt>
          <dd className="text-sm text-ink">{codes.join(", ")}</dd>
        </div>
      )}
    </dl>
  );
}

function Progress({ received, total }) {
  const pct = total ? Math.min(100, Math.round((received / total) * 100)) : null;
  return (
    <div role="status" aria-live="polite" className="space-y-2">
      <div className="flex items-center gap-2 text-sm text-ink-2">
        <span className="h-4 w-4 animate-spin rounded-full border-2 border-line border-t-accent" />
        {received === null ? "Fetching complete details…" : `Downloading PDF… ${mb(received)}${total ? ` of ${mb(total)} (${pct}%)` : ""}`}
      </div>
      {received !== null && (
        <div className="h-1.5 overflow-hidden rounded-full bg-surface-2">
          <div className={`h-full rounded-full bg-accent ${pct === null ? "w-1/3 animate-pulse" : ""}`} style={pct === null ? undefined : { width: `${pct}%` }} />
        </div>
      )}
    </div>
  );
}

// "Load complete details" for the current document: its full metadata plus the PDF where the
// server can fetch it. The last 5 PDFs stay in memory for this visit (see lib/docCache.js).
export default function FullDetails({ id, codes }) {
  const [pdf, setPdf] = useState(() => pdfCache.get(id) ?? null);
  const [details, setDetails] = useState(() => {
    const cached = detailsCache.get(id) ?? null;
    return cached?.embeddable && !pdf ? null : cached; // its PDF was removed: load it all again
  });
  const [fromCache] = useState(() => Boolean(details));
  const [status, setStatus] = useState(details ? "done" : "idle"); // idle | details | pdf | done | error
  const [progress, setProgress] = useState({ received: null, total: null });
  const [pdfFailed, setPdfFailed] = useState(false);
  const [message, setMessage] = useState("");
  const abort = useRef(null);
  const viewer = useRef(null);

  useEffect(() => () => abort.current?.abort(), []); // stop the download when leaving the page

  const load = async () => {
    const ctrl = new AbortController();
    abort.current = ctrl;
    setStatus("details");
    setPdfFailed(false);
    setProgress({ received: null, total: null });
    try {
      const d = await api.fullText(id);
      detailsCache.set(id, d);
      setDetails(d);
      if (d.embeddable) {
        setStatus("pdf");
        setProgress({ received: 0, total: null });
        try {
          const url = await downloadPdf(api.pdfUrl(id), (received, total) => setProgress({ received, total }), ctrl.signal);
          pdfCache.set(id, url);
          setPdf(url);
        } catch (e) {
          if (e.name === "AbortError") return;
          setPdfFailed(true);
        }
      }
      setStatus("done");
    } catch (e) {
      if (e.name === "AbortError") return;
      setMessage(e.message.replace(/^\d+: /, ""));
      setStatus("error");
    }
  };

  const title = details?.kind === "patent" ? "Complete details" : "Complete details and full paper";
  const busy = status === "details" || status === "pdf";
  const openLabel = details?.kind === "patent" ? "View full patent" : "Read full paper";

  return (
    <Card
      title={title}
      subtitle={fromCache ? `Loaded from this visit's saved documents (up to ${PDF_CACHE_LIMIT} recent PDFs are kept until you close the site)` : undefined}
      action={pdf && (
        <div className="flex items-center gap-3">
          <button onClick={() => viewer.current?.requestFullscreen?.()} className="rounded-md border border-line px-2 py-1 text-xs font-medium text-ink-2 hover:border-accent hover:bg-surface-2 hover:text-ink">
            ⛶ Full screen
          </button>
          <a href={pdf} target="_blank" rel="noreferrer" className="btn-link text-xs font-medium text-accent">Open in new tab ↗</a>
        </div>
      )}
    >
      <div className="space-y-4">
        {status === "idle" && (
          <div className="flex flex-wrap items-center gap-3">
            <p className="flex-1 text-sm text-ink-2">Fetch the authors, journal, citations and the full paper where available.</p>
            <button onClick={load} className="btn-primary rounded-md px-3 py-1.5 text-sm font-medium">Load complete details</button>
          </div>
        )}
        {status === "error" && (
          <div className="flex flex-wrap items-center gap-3">
            <p className="flex-1 text-sm text-critical">{message}</p>
            <button onClick={load} className="btn-primary rounded-md px-3 py-1.5 text-sm font-medium">Try again</button>
          </div>
        )}
        {details && <Facts d={details} codes={codes} />}
        {busy && <Progress received={progress.received} total={progress.total} />}
        {pdf && <iframe ref={viewer} src={`${pdf}#view=FitH`} title="Full paper PDF" allowFullScreen className="h-[85vh] w-full rounded-md border border-line bg-white" />}
        {status === "done" && !pdf && details?.url && (
          <div className="flex flex-wrap items-center gap-3 border-t border-line pt-3">
            <p className="flex-1 text-sm text-ink-2">
              {details.kind === "patent" ? "The complete patent, with claims and drawings, is on Google Patents."
                : pdfFailed ? "The PDF could not be loaded here, but you can read it at the source."
                : details.is_open_access ? "This paper is free to read on the publisher's site."
                : "This paper is on the publisher's site and may need a subscription."}
            </p>
            <a href={details.url} target="_blank" rel="noreferrer" className="btn-primary inline-block rounded-md px-3 py-1.5 text-sm font-medium">{openLabel} ↗</a>
          </div>
        )}
        {status === "done" && !details?.url && <p className="text-sm text-muted">No online copy was found for this document.</p>}
      </div>
    </Card>
  );
}
