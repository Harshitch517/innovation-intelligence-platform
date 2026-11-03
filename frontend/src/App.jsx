import { lazy, Suspense } from "react";
import { NavLink, Route, Routes } from "react-router-dom";
import { Loading } from "./components/ui";
import Search from "./pages/Search";

// Pages load on demand, so the charting and graph libraries are only fetched when needed.
const Cluster = lazy(() => import("./pages/Cluster"));
const Compare = lazy(() => import("./pages/Compare"));
const Dashboard = lazy(() => import("./pages/Dashboard"));
const Document = lazy(() => import("./pages/Document"));
const Institutions = lazy(() => import("./pages/Institutions"));
const Network = lazy(() => import("./pages/Network"));

const NAV = [
  { to: "/", label: "Search", end: true },
  { to: "/dashboard", label: "Dashboard" },
  { to: "/compare", label: "Compare" },
  { to: "/institutions", label: "Institutions" },
  { to: "/network", label: "Network" },
];

export default function App() {
  return (
    <div className="min-h-screen">
      <header className="sticky top-0 z-20 border-b border-line bg-surface/90 backdrop-blur">
        <div className="mx-auto flex max-w-6xl flex-wrap items-center gap-x-6 gap-y-2 px-4 py-3">
          <NavLink to="/" className="flex items-center gap-2 font-semibold text-ink">
            <svg width="22" height="22" viewBox="0 0 32 32" aria-hidden>
              <circle cx="9" cy="16" r="5" fill="var(--d-ai)" />
              <circle cx="23" cy="9" r="4" fill="var(--d-hc)" />
              <circle cx="23" cy="23" r="4" fill="var(--d-rb)" />
              <path d="M13 14l6-3M13 18l6 3" stroke="var(--ink-2)" strokeWidth="2" />
            </svg>
            Innovation Intelligence
          </NavLink>
          <nav className="flex flex-wrap gap-1 text-sm">
            {NAV.map((n) => (
              <NavLink
                key={n.to}
                to={n.to}
                end={n.end}
                className={({ isActive }) =>
                  `rounded-md px-2.5 py-1.5 ${isActive ? "bg-surface-2 font-medium text-ink" : "text-ink-2 hover:text-ink"}`
                }
              >
                {n.label}
              </NavLink>
            ))}
          </nav>
        </div>
      </header>
      <main className="mx-auto max-w-6xl px-4 py-6">
        <Suspense fallback={<Loading />}>
        <Routes>
          <Route path="/" element={<Search />} />
          <Route path="/documents/:id" element={<Document />} />
          <Route path="/compare" element={<Compare />} />
          <Route path="/dashboard" element={<Dashboard />} />
          <Route path="/institutions" element={<Institutions />} />
          <Route path="/network" element={<Network />} />
          <Route path="/clusters/:id" element={<Cluster />} />
          <Route path="*" element={<p className="text-ink-2">Page not found.</p>} />
        </Routes>
        </Suspense>
      </main>
      <footer className="mx-auto max-w-6xl px-4 pb-8 text-xs text-muted">
        INT395 · 31,169 university patents (Google Patents) and papers (OpenAlex), 2020–2025, including an India supplement
      </footer>
    </div>
  );
}
