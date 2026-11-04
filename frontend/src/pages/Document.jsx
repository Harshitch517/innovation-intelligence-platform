import { useState } from "react";
import { Link, useParams } from "react-router-dom";
import { api } from "../api";
import FullDetails from "../components/FullDetails";
import { Card, DocCard, DomainBadge, Empty, ErrorBox, Explain, Loading, SourceBadge } from "../components/ui";
import { useApi } from "../lib/useApi";

function ClusterChip({ cluster, label }) {
  if (!cluster) return null;
  return (
    <Link to={`/clusters/${encodeURIComponent(cluster.cluster_id)}`} className="block rounded-lg border border-line p-3 hover:bg-surface-2">
      <p className="text-xs text-muted">{label} · {cluster.cluster_id}</p>
      <p className="mt-0.5 text-sm font-medium text-ink">{cluster.name || cluster.keywords.slice(0, 4).join(", ")}</p>
      {cluster.name && <p className="mt-0.5 text-xs text-muted">{cluster.keywords.slice(0, 6).join(", ")}</p>}
    </Link>
  );
}

export default function Document() {
  const { id } = useParams();
  const [otherOnly, setOtherOnly] = useState(true);
  const doc = useApi(() => api.document(id), [id]);
  const sim = useApi(() => api.similar(id, { k: 8, other_institutions_only: otherOnly }), [id, otherOnly]);

  if (doc.error) return <ErrorBox error={doc.error} />;
  if (!doc.data) return <Loading />;
  const d = doc.data;
  return (
    <div className="grid gap-5 lg:grid-cols-[minmax(0,3fr)_minmax(0,2fr)]">
      <div className="space-y-4">
        <div className="flex flex-wrap items-center gap-2">
          <SourceBadge source={d.source} />
          <DomainBadge domain={d.domain} />
          <span className="tabular text-sm text-muted">{d.year}</span>
          <span className="text-xs text-muted">{d.doc_id}</span>
        </div>
        <h1 className="text-xl font-semibold leading-snug sm:text-2xl">{d.title}</h1>
        <p className="text-sm text-ink-2">
          <Link to={`/institutions?ids=${encodeURIComponent(d.institution)}`} className="font-medium hover:underline">{d.institution_name}</Link>
          {d.country && <span className="text-muted"> · {d.country}</span>}
        </p>
        <Card title="Abstract">
          <p className="text-sm leading-relaxed text-ink-2">{d.abstract}</p>
        </Card>
        <FullDetails key={id} id={id} codes={d.raw_codes.slice(0, 12)} />
        <div className="grid gap-3 sm:grid-cols-2">
          <ClusterChip cluster={d.cluster} label="Sub-area" />
          <ClusterChip cluster={d.global_cluster} label="Global cluster" />
        </div>
        <p className="text-xs text-muted">
          {d.source === "patent" ? "CPC codes" : "OpenAlex topics"}: {d.raw_codes.slice(0, 8).join(", ")}
        </p>
      </div>

      <div className="space-y-3">
        <div className="flex items-center justify-between">
          <h2 className="text-sm font-semibold">Similar innovations</h2>
          <label className="flex items-center gap-2 text-xs text-ink-2">
            <input type="checkbox" checked={otherOnly} onChange={(e) => setOtherOnly(e.target.checked)} />
            other institutions only
          </label>
        </div>
        {sim.error && <ErrorBox error={sim.error} />}
        {!sim.data && !sim.error && <Loading />}
        {sim.data?.results.length === 0 && <Empty>No similar documents.</Empty>}
        <div className={`space-y-3 ${sim.loading ? "opacity-50" : ""}`}>
          {sim.data?.results.map((s) => (
            <DocCard key={s.doc_id} doc={s}>
              <div className="mt-2 flex flex-wrap items-center gap-3">
                <Link to={`/compare?a=${encodeURIComponent(d.doc_id)}&b=${encodeURIComponent(s.doc_id)}`} className="text-xs font-medium text-accent hover:underline">
                  Compare
                </Link>
                <Explain a={d.doc_id} b={s.doc_id} />
              </div>
            </DocCard>
          ))}
        </div>
      </div>
    </div>
  );
}
