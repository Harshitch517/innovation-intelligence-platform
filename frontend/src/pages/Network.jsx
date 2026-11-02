import cytoscape from "cytoscape";
import { useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../api";
import { Card, DomainBadge, ErrorBox, Loading, Select, domainOptions } from "../components/ui";
import { DOMAINS, fmt, useColors } from "../lib/theme";
import { useApi } from "../lib/useApi";

const SHAPE = Object.fromEntries(DOMAINS.map((d) => [d.name, d.shape]));

export default function Network() {
  const c = useColors();
  const [domain, setDomain] = useState("");
  const [minSim, setMinSim] = useState(0.8);
  const [selected, setSelected] = useState(null);
  const box = useRef(null);
  const { data, error, loading } = useApi(() => api.network({ domain: domain || undefined, min_similarity: minSim }), [domain, minSim]);

  useEffect(() => {
    if (!data || !box.current) return;
    const linked = new Set(data.edges.flatMap((e) => [e.source, e.target]));
    const nodes = data.nodes.filter((n) => linked.has(n.id));
    const cy = cytoscape({
      container: box.current,
      elements: [
        ...nodes.map((n) => ({ data: { ...n, size: 10 + Math.sqrt(n.documents) * 3 } })),
        ...data.edges.map((e, i) => ({ data: { id: `e${i}`, ...e } })),
      ],
      style: [
        {
          selector: "node",
          style: {
            width: "data(size)", height: "data(size)",
            "background-color": (n) => c.domain[n.data("top_domain")],
            shape: (n) => SHAPE[n.data("top_domain")] || "ellipse",
            "border-width": 1.5, "border-color": c.surface,
            label: (n) => (n.data("documents") >= 60 ? n.data("name") : ""),
            "font-size": 9, color: c.ink2, "text-valign": "bottom", "text-margin-y": 3,
            "text-outline-color": c.surface, "text-outline-width": 2,
          },
        },
        { selector: "edge", style: { width: (e) => 0.5 + (e.data("similarity") - 0.7) * 6, "line-color": c.muted, opacity: 0.25, "curve-style": "haystack" } },
        { selector: "node:selected", style: { "border-color": c.ink, "border-width": 3 } },
        { selector: ".faded", style: { opacity: 0.12 } },
      ],
      layout: { name: "cose", animate: false, randomize: true, nodeRepulsion: () => 6000, idealEdgeLength: () => 60, numIter: 1500, fit: true, padding: 20 },
      wheelSensitivity: 0.3,
    });
    cy.on("tap", "node", (evt) => {
      const n = evt.target;
      cy.elements().addClass("faded");
      n.closedNeighborhood().removeClass("faded");
      setSelected({ ...n.data(), neighbours: n.neighborhood("node").map((m) => m.data()) });
    });
    cy.on("tap", (evt) => {
      if (evt.target === cy) { cy.elements().removeClass("faded"); setSelected(null); }
    });
    return () => cy.destroy();
  }, [data, c]);

  return (
    <div className="space-y-5">
      <div>
        <h1 className="text-2xl font-semibold">Institution network</h1>
        <p className="mt-1 text-sm text-ink-2">
          Institutions linked by similar research profiles (mean embedding similarity) or shared sub-areas. Colour and shape show each institution's main domain.
        </p>
      </div>
      <div className="flex flex-wrap items-end gap-4">
        <Select label="Domain" value={domain} onChange={(v) => { setDomain(v); setSelected(null); }} options={domainOptions} />
        <label className="flex flex-col gap-1 text-xs text-muted">
          Minimum similarity: <span className="tabular text-ink">{minSim.toFixed(2)}</span>
          <input type="range" min="0.6" max="0.95" step="0.01" value={minSim} onChange={(e) => setMinSim(Number(e.target.value))} />
        </label>
        {data && <p className="tabular text-xs text-muted">{fmt(data.edges.length)} links · {fmt(data.nodes.length)} institutions</p>}
      </div>
      {error && <ErrorBox error={error} />}
      <div className="grid gap-4 lg:grid-cols-[minmax(0,3fr)_minmax(0,1fr)]">
        <div className="relative overflow-hidden rounded-xl border border-line bg-surface">
          {!data && !error && <div className="absolute inset-0 grid place-items-center"><Loading label="Laying out network…" /></div>}
          <div ref={box} className={`h-[520px] w-full sm:h-[620px] ${loading ? "opacity-50" : ""}`} aria-label="Institution network graph" role="img" />
          <ul className="flex flex-wrap gap-3 border-t border-line px-3 py-2 text-xs text-ink-2">
            {DOMAINS.map((d) => (
              <li key={d.name} className="flex items-center gap-1.5">
                <svg width="12" height="12" aria-hidden>
                  {d.shape === "ellipse" && <circle cx="6" cy="6" r="5" fill={c.domain[d.name]} />}
                  {d.shape === "round-rectangle" && <rect x="1" y="1" width="10" height="10" rx="2" fill={c.domain[d.name]} />}
                  {d.shape === "triangle" && <polygon points="6,1 11,11 1,11" fill={c.domain[d.name]} />}
                  {d.shape === "diamond" && <polygon points="6,0 12,6 6,12 0,6" fill={c.domain[d.name]} />}
                  {d.shape === "hexagon" && <polygon points="3,1 9,1 12,6 9,11 3,11 0,6" fill={c.domain[d.name]} />}
                </svg>
                {d.name}
              </li>
            ))}
          </ul>
        </div>
        <Card title={selected ? selected.name : "Select an institution"} subtitle={selected ? `${selected.country || ""} · ${fmt(selected.documents)} documents` : "Click a node to see its links"}>
          {selected && (
            <div className="space-y-3">
              <DomainBadge domain={selected.top_domain} />
              <Link to={`/institutions?ids=${encodeURIComponent(selected.id)}`} className="block text-sm font-medium text-accent hover:underline">Open profile →</Link>
              <p className="text-xs font-medium text-muted">Linked institutions ({selected.neighbours.length})</p>
              <ul className="max-h-80 space-y-1 overflow-auto text-sm text-ink-2">
                {selected.neighbours.sort((a, b) => b.documents - a.documents).map((n) => <li key={n.id}>{n.name}</li>)}
              </ul>
            </div>
          )}
        </Card>
      </div>
    </div>
  );
}
