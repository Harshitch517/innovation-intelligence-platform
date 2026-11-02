import { useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { Bar, BarChart, CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { api } from "../api";
import { Card, ChartCard, ChartTooltip, DomainBadge, ErrorBox, Loading, Select, StatTile, domainOptions } from "../components/ui";
import { DOMAINS, fmt, pct, useColors } from "../lib/theme";
import { useApi } from "../lib/useApi";

const YEARS = [2020, 2021, 2022, 2023, 2024, 2025];

function DomainSourceChart({ data, c }) {
  const rows = DOMAINS.map((d) => {
    const r = { domain: d.short, full: d.name, patent: 0, paper: 0 };
    data.by_domain_year.filter((x) => x.domain === d.name).forEach((x) => (r[x.source] += x.count));
    return r;
  });
  return (
    <ChartCard title="Most researched domains" subtitle="Documents per domain, patents and papers"
      columns={["Domain", "Patents", "Papers", "Total"]} rows={rows.map((r) => [r.full, r.patent, r.paper, r.patent + r.paper])}>
      <ResponsiveContainer width="100%" height={240}>
        <BarChart data={rows} layout="vertical" margin={{ left: 8, right: 16 }} barCategoryGap={10}>
          <CartesianGrid horizontal={false} />
          <XAxis type="number" tickFormatter={fmt} />
          <YAxis type="category" dataKey="domain" width={78} />
          <Tooltip content={<ChartTooltip />} cursor={{ fill: c.grid, opacity: 0.4 }} />
          <Legend iconType="rect" itemSorter={null} />
          <Bar dataKey="patent" name="Patents" stackId="s" fill={c.patent} stroke={c.surface} strokeWidth={2} />
          <Bar dataKey="paper" name="Papers" stackId="s" fill={c.paper} stroke={c.surface} strokeWidth={2} radius={[0, 4, 4, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </ChartCard>
  );
}

function YearChart({ data, c }) {
  const rows = YEARS.map((y) => {
    const r = { year: y };
    DOMAINS.forEach((d) => {
      r[d.name] = data.by_domain_year.filter((x) => x.domain === d.name && x.year === y).reduce((s, x) => s + x.count, 0);
    });
    return r;
  });
  return (
    <ChartCard title="Innovation over time" subtitle="Documents per year by domain (patents granted, papers published)"
      columns={["Year", ...DOMAINS.map((d) => d.short)]} rows={rows.map((r) => [r.year, ...DOMAINS.map((d) => r[d.name])])}>
      <ResponsiveContainer width="100%" height={240}>
        <LineChart data={rows} margin={{ left: 0, right: 16, top: 8 }}>
          <CartesianGrid vertical={false} />
          <XAxis dataKey="year" />
          <YAxis tickFormatter={fmt} width={44} />
          <Tooltip content={<ChartTooltip />} cursor={{ stroke: c.muted, strokeWidth: 1 }} />
          <Legend iconType="plainline" itemSorter={null} />
          {DOMAINS.map((d) => (
            <Line key={d.name} dataKey={d.name} name={d.short} stroke={c.domain[d.name]} strokeWidth={2} dot={false}
              activeDot={{ r: 4, stroke: c.surface, strokeWidth: 2 }} />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </ChartCard>
  );
}

function TopInstitutions({ data, c }) {
  const rows = data.top_institutions.slice(0, 10).map((t) => ({ ...t, label: t.name.length > 30 ? `${t.name.slice(0, 28)}…` : t.name }));
  return (
    <ChartCard title="Top institutions" subtitle="Documents (patents + papers)" columns={["Institution", "Documents"]}
      rows={rows.map((r) => [r.name, r.count])}>
      <ResponsiveContainer width="100%" height={300}>
        <BarChart data={rows} layout="vertical" margin={{ left: 8, right: 24 }} barCategoryGap={6}>
          <CartesianGrid horizontal={false} />
          <XAxis type="number" tickFormatter={fmt} />
          <YAxis type="category" dataKey="label" width={200} tick={{ fontSize: 11 }} interval={0} />
          <Tooltip content={<ChartTooltip labelFormatter={(l) => rows.find((r) => r.label === l)?.name || l} />} cursor={{ fill: c.grid, opacity: 0.4 }} />
          <Bar dataKey="count" name="documents" fill={c.accent} radius={[0, 4, 4, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </ChartCard>
  );
}

function Sparkline({ counts, color }) {
  const values = YEARS.map((y) => counts[y] ?? 0);
  const max = Math.max(...values, 1);
  const pts = values.map((v, i) => `${(i / (values.length - 1)) * 76 + 2},${22 - (v / max) * 18}`).join(" ");
  return (
    <svg width="80" height="24" aria-label={`yearly counts ${values.join(", ")}`}>
      <polyline points={pts} fill="none" stroke={color} strokeWidth="2" strokeLinejoin="round" />
    </svg>
  );
}

function Emerging({ c }) {
  const [domain, setDomain] = useState("");
  const { data, error, loading } = useApi(() => api.trends(domain || undefined), [domain]);
  return (
    <Card title="Emerging technologies" subtitle="Sub-areas ranked by growth, 2024–25 vs 2022–23 (at least 40 documents)"
      action={<Select label="" value={domain} onChange={setDomain} options={domainOptions} />}>
      {error && <ErrorBox error={error} />}
      {!data && !error && <Loading />}
      {data && (
        <ol className={`divide-y divide-line ${loading ? "opacity-50" : ""}`}>
          {data.emerging.map((e) => (
            <li key={e.cluster_id} className="flex flex-wrap items-center gap-x-4 gap-y-1 py-2.5">
              <span className="tabular w-6 text-sm text-muted">{e.emerging_rank}</span>
              <div className="min-w-0 flex-1">
                <Link to={`/clusters/${encodeURIComponent(e.cluster_id)}`} className="text-sm font-medium text-ink hover:text-accent">
                  {e.name || e.keywords.slice(0, 4).join(", ")}
                </Link>
                <div className="mt-0.5 flex flex-wrap items-center gap-2 text-xs text-muted">
                  <DomainBadge domain={e.domain} /> {fmt(e.size)} documents
                </div>
              </div>
              <Sparkline counts={e.counts} color={c.domain[e.domain]} />
              <span className="tabular w-16 text-right text-sm font-semibold text-good">{pct(e.recent_growth)}</span>
            </li>
          ))}
        </ol>
      )}
    </Card>
  );
}

function Classify({ c }) {
  const [title, setTitle] = useState("");
  const [abstract, setAbstract] = useState("");
  const [state, setState] = useState({ status: "idle" });
  const run = (e) => {
    e.preventDefault();
    setState({ status: "loading" });
    api.classify(title, abstract).then((r) => setState({ status: "done", ...r })).catch((err) => setState({ status: "error", error: err }));
  };
  return (
    <Card title="Classify a new innovation" subtitle="Paste a title and abstract; the served model (MiniLM + Logistic Regression) predicts the domain">
      <form onSubmit={run} className="space-y-2">
        <input value={title} onChange={(e) => setTitle(e.target.value)} placeholder="Title" required minLength={3}
          className="w-full rounded-md border border-line bg-surface px-2.5 py-1.5 text-sm" />
        <textarea value={abstract} onChange={(e) => setAbstract(e.target.value)} placeholder="Abstract (at least 20 characters)" required minLength={20} rows={4}
          className="w-full rounded-md border border-line bg-surface px-2.5 py-1.5 text-sm" />
        <button className="rounded-md bg-accent px-3 py-1.5 text-sm font-medium text-white hover:opacity-90" disabled={state.status === "loading"}>
          {state.status === "loading" ? "Classifying…" : "Classify"}
        </button>
      </form>
      {state.status === "error" && <p className="mt-2 text-sm text-critical">{state.error.message}</p>}
      {state.status === "done" && (
        <div className="mt-3 space-y-1.5">
          <p className="text-sm">Predicted: <strong>{state.domain}</strong></p>
          {Object.entries(state.probabilities).map(([d, p]) => (
            <div key={d} className="flex items-center gap-2 text-xs">
              <span className="w-36 text-ink-2">{d}</span>
              <span className="h-2 flex-1 overflow-hidden rounded-full bg-surface-2">
                <span className="block h-full rounded-full" style={{ width: `${p * 100}%`, background: c.domain[d] }} />
              </span>
              <span className="tabular w-12 text-right text-ink">{(p * 100).toFixed(1)}%</span>
            </div>
          ))}
        </div>
      )}
    </Card>
  );
}

export default function Dashboard() {
  const c = useColors();
  const { data, error } = useApi(() => api.overview(), []);
  const totals = useMemo(() => data && [
    ["Documents", data.documents], ["Patents", data.patents], ["Papers", data.papers], ["Institutions", data.institutions],
  ], [data]);
  if (error) return <ErrorBox error={error} />;
  if (!data) return <Loading />;
  return (
    <div className="space-y-5">
      <div>
        <h1 className="text-2xl font-semibold">Dashboard</h1>
        <p className="mt-1 text-sm text-ink-2">University and research-institute innovation, 2020–2025, across five technology domains.</p>
      </div>
      <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
        {totals.map(([l, v]) => <StatTile key={l} label={l} value={v} />)}
      </div>
      <div className="grid gap-4 lg:grid-cols-2">
        <DomainSourceChart data={data} c={c} />
        <YearChart data={data} c={c} />
      </div>
      <div className="grid gap-4 lg:grid-cols-2">
        <Emerging c={c} />
        <TopInstitutions data={data} c={c} />
      </div>
      <Classify c={c} />
    </div>
  );
}
