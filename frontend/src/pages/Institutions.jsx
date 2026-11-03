import { useEffect, useState } from "react";
import { Link, useSearchParams } from "react-router-dom";
import { Bar, BarChart, CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { api } from "../api";
import { Card, ChartCard, ChartTooltip, DomainBadge, Empty, ErrorBox, Loading } from "../components/ui";
import { DOMAINS, fmt, useColors } from "../lib/theme";

const YEARS = ["2020", "2021", "2022", "2023", "2024", "2025"];
const SEP = "|";

function Picker({ onPick, disabled }) {
  const [q, setQ] = useState("");
  const [list, setList] = useState([]);
  useEffect(() => {
    const t = setTimeout(() => api.institutions({ q: q || undefined, limit: 12 }).then(setList).catch(() => setList([])), 250);
    return () => clearTimeout(t);
  }, [q]);
  return (
    <Card title="Add an institution" subtitle="Institutions with at least 10 documents; compare up to three">
      <input value={q} onChange={(e) => setQ(e.target.value)} placeholder="Search by name, e.g. Tsinghua" disabled={disabled}
        aria-label="Search institutions" className="w-full rounded-md border border-line bg-surface px-2.5 py-1.5 text-sm" />
      <ul className="mt-2 max-h-64 divide-y divide-line overflow-auto">
        {list.map((i) => (
          <li key={i.institution}>
            <button onClick={() => onPick(i.institution)} disabled={disabled}
              className="row-btn flex w-full items-center justify-between gap-2 rounded px-1 py-2 text-left text-sm hover:text-accent">
              <span>{i.name} <span className="text-xs text-muted">{i.country}</span></span>
              <span className="tabular text-xs text-muted">{fmt(i.documents)}</span>
            </button>
          </li>
        ))}
      </ul>
    </Card>
  );
}

export default function Institutions() {
  const c = useColors();
  const [params, setParams] = useSearchParams();
  const ids = (params.get("ids") || "").split(SEP).filter(Boolean).slice(0, 3);
  const [details, setDetails] = useState({});
  const [error, setError] = useState(null);
  const setIds = (next) => setParams(next.length ? { ids: next.join(SEP) } : {});

  useEffect(() => {
    ids.filter((id) => !details[id]).forEach((id) =>
      api.institution(id).then((d) => setDetails((s) => ({ ...s, [id]: d }))).catch(setError));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ids.join(SEP)]);

  const chosen = ids.map((id) => details[id]).filter(Boolean);
  const colorOf = (inst) => c.compare[ids.indexOf(inst)];
  const domainRows = DOMAINS.map((d) => Object.fromEntries([["domain", d.short], ...chosen.map((x) => [x.name, x.by_domain[d.name] || 0])]));
  const yearRows = YEARS.map((y) => Object.fromEntries([["year", y], ...chosen.map((x) => [x.name, x.by_year[y] || 0])]));

  return (
    <div className="space-y-5">
      <div>
        <h1 className="text-2xl font-semibold">Institutions</h1>
        <p className="mt-1 text-sm text-ink-2">Compare research profiles: domain mix, output over time, sub-areas and similar institutions.</p>
      </div>
      {error && <ErrorBox error={error} />}
      <div className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_minmax(0,2fr)]">
        <Picker onPick={(id) => !ids.includes(id) && setIds([...ids, id])} disabled={ids.length >= 3} />
        <div className="grid gap-3 sm:grid-cols-3">
          {ids.length === 0 && <Card className="sm:col-span-3"><Empty>Pick up to three institutions to compare.</Empty></Card>}
          {ids.map((id, i) => {
            const d = details[id];
            return (
              <Card key={id}>
                {!d ? <Loading /> : (
                  <div className="space-y-2">
                    <div className="flex items-start justify-between gap-2">
                      <div className="flex items-start gap-2">
                        <span className="mt-1.5 h-2.5 w-2.5 shrink-0 rounded-sm" style={{ background: colorOf(id) }} aria-hidden />
                        <h2 className="text-sm font-semibold leading-snug">{d.name}</h2>
                      </div>
                      <button onClick={() => setIds(ids.filter((x) => x !== id))} className="btn-link rounded px-1 text-xs text-muted hover:text-critical" aria-label={`Remove ${d.name}`}>✕</button>
                    </div>
                    <p className="text-xs text-muted">{d.country} · main domain</p>
                    <DomainBadge domain={d.top_domain} />
                    <dl className="tabular grid grid-cols-3 gap-1 pt-1 text-center">
                      {[["docs", d.documents], ["patents", d.patents], ["papers", d.papers]].map(([k, v]) => (
                        <div key={k} className="rounded-md bg-surface-2 py-1.5"><dt className="text-[11px] text-muted">{k}</dt><dd className="text-sm font-semibold">{fmt(v)}</dd></div>
                      ))}
                    </dl>
                  </div>
                )}
              </Card>
            );
          })}
        </div>
      </div>

      {chosen.length > 0 && (
        <>
          <div className="grid gap-4 lg:grid-cols-2">
            <ChartCard title="Domain mix" subtitle="Documents per domain" columns={["Domain", ...chosen.map((x) => x.name)]}
              rows={domainRows.map((r) => [r.domain, ...chosen.map((x) => r[x.name])])}>
              <ResponsiveContainer width="100%" height={250}>
                <BarChart data={domainRows} margin={{ left: 0, right: 8 }} barGap={2}>
                  <CartesianGrid vertical={false} />
                  <XAxis dataKey="domain" tick={{ fontSize: 11 }} />
                  <YAxis width={36} />
                  <Tooltip content={<ChartTooltip />} cursor={{ fill: c.grid, opacity: 0.4 }} />
                  <Legend iconType="rect" itemSorter={null} />
                  {chosen.map((x) => <Bar key={x.institution} dataKey={x.name} fill={colorOf(x.institution)} radius={[4, 4, 0, 0]} />)}
                </BarChart>
              </ResponsiveContainer>
            </ChartCard>
            <ChartCard title="Output over time" subtitle="Documents per year" columns={["Year", ...chosen.map((x) => x.name)]}
              rows={yearRows.map((r) => [r.year, ...chosen.map((x) => r[x.name])])}>
              <ResponsiveContainer width="100%" height={250}>
                <LineChart data={yearRows} margin={{ left: 0, right: 16, top: 8 }}>
                  <CartesianGrid vertical={false} />
                  <XAxis dataKey="year" />
                  <YAxis width={36} />
                  <Tooltip content={<ChartTooltip />} cursor={{ stroke: c.muted, strokeWidth: 1 }} />
                  <Legend iconType="plainline" itemSorter={null} />
                  {chosen.map((x, i) => (
                    <Line key={x.institution} dataKey={x.name} stroke={colorOf(x.institution)} strokeWidth={2} dot={{ r: 4, strokeWidth: 0, fill: colorOf(x.institution) }}
                      activeDot={{ r: 5, stroke: c.surface, strokeWidth: 2 }} />
                  ))}
                </LineChart>
              </ResponsiveContainer>
            </ChartCard>
          </div>
          <div className="grid gap-4 md:grid-cols-3">
            {chosen.map((x, i) => (
              <Card key={x.institution} title={x.name} subtitle="Most similar institutions (research profile)">
                <ul className="space-y-1.5">
                  {x.similar.slice(0, 6).map((s) => (
                    <li key={s.institution} className="flex items-center justify-between gap-2 text-sm">
                      <button onClick={() => ids.length < 3 && !ids.includes(s.institution) && setIds([...ids, s.institution])}
                        className="btn-link rounded text-left text-ink-2 hover:text-accent">{s.name}</button>
                      <span className="tabular text-xs text-muted">{s.similarity.toFixed(2)}</span>
                    </li>
                  ))}
                </ul>
                <p className="mt-4 text-xs font-medium text-muted">Top sub-areas</p>
                <ul className="mt-1 space-y-1">
                  {x.top_clusters.map((cl) => (
                    <li key={cl.cluster_id}>
                      <Link to={`/clusters/${encodeURIComponent(cl.cluster_id)}`} className="text-sm text-ink-2 hover:text-accent">
                        {cl.name || cl.keywords.slice(0, 3).join(", ")}
                      </Link>
                    </li>
                  ))}
                </ul>
                <span className="sr-only">series colour {i + 1}</span>
              </Card>
            ))}
          </div>
        </>
      )}
    </div>
  );
}
