import { Link, useParams } from "react-router-dom";
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { api } from "../api";
import { Card, ChartCard, ChartTooltip, DocCard, DomainBadge, ErrorBox, Loading } from "../components/ui";
import { fmt, pct, useColors } from "../lib/theme";
import { useApi } from "../lib/useApi";

export default function Cluster() {
  const { id } = useParams();
  const c = useColors();
  const { data, error } = useApi(() => api.cluster(id), [id]);
  if (error) return <ErrorBox error={error} />;
  if (!data) return <Loading />;
  const rows = Object.entries(data.counts).map(([year, count]) => ({ year, count }));
  const color = data.domain ? c.domain[data.domain] : c.accent;
  return (
    <div className="space-y-5">
      <div>
        <p className="text-xs text-muted">{data.scope === "global" ? "Global cluster" : "Sub-area"} · {data.cluster_id}</p>
        <h1 className="mt-1 text-2xl font-semibold">{data.name || data.keywords.slice(0, 4).join(", ")}</h1>
        <div className="mt-2 flex flex-wrap items-center gap-3 text-sm text-ink-2">
          {data.domain && <DomainBadge domain={data.domain} />}
          <span>{fmt(data.size)} documents</span>
          {data.recent_growth != null && <span>growth <strong className="text-good">{pct(data.recent_growth)}</strong> (2024–25 vs 2022–23)</span>}
          {data.emerging_rank && <span>emerging rank #{data.emerging_rank}</span>}
        </div>
        {data.summary && <p className="mt-3 max-w-3xl text-sm leading-relaxed text-ink-2">{data.summary}</p>}
        {data.emerging_summary && (
          <p className="mt-3 max-w-3xl rounded-lg border border-line bg-surface p-3 text-sm leading-relaxed text-ink-2">
            <span className="mb-1 block text-xs font-medium text-muted">Why it is emerging (Gemini)</span>
            {data.emerging_summary}
          </p>
        )}
        <p className="mt-2 text-xs text-muted">Keywords: {data.keywords.join(", ")}</p>
      </div>
      <div className="grid gap-4 lg:grid-cols-2">
        <ChartCard title="Documents per year" columns={["Year", "Documents"]} rows={rows.map((r) => [r.year, r.count])}>
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={rows} margin={{ left: 0, right: 8 }}>
              <CartesianGrid vertical={false} />
              <XAxis dataKey="year" />
              <YAxis width={36} />
              <Tooltip content={<ChartTooltip />} cursor={{ fill: c.grid, opacity: 0.4 }} />
              <Bar dataKey="count" name="documents" fill={color} radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </ChartCard>
        <Card title="Leading institutions">
          <ul className="space-y-1.5">
            {data.top_institutions.map((t) => (
              <li key={t.institution} className="flex justify-between gap-2 text-sm">
                <Link to={`/institutions?ids=${encodeURIComponent(t.institution)}`} className="text-ink-2 hover:text-accent">{t.name}</Link>
                <span className="tabular text-xs text-muted">{t.count}</span>
              </li>
            ))}
          </ul>
        </Card>
      </div>
      <div className="space-y-3">
        <h2 className="text-sm font-semibold">Recent documents</h2>
        {data.members.map((m) => <DocCard key={m.doc_id} doc={m} />)}
      </div>
    </div>
  );
}
