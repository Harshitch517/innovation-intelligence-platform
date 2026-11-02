import { useEffect, useState } from "react";

// Domains in fixed categorical order (colour follows the domain everywhere in the app).
export const DOMAINS = [
  { name: "Artificial Intelligence", short: "AI", token: "--d-ai", shape: "ellipse" },
  { name: "Healthcare", short: "Healthcare", token: "--d-hc", shape: "round-rectangle" },
  { name: "Robotics", short: "Robotics", token: "--d-rb", shape: "triangle" },
  { name: "Cybersecurity", short: "Cyber", token: "--d-cs", shape: "diamond" },
  { name: "Renewable Energy", short: "Renewable", token: "--d-re", shape: "hexagon" },
];
export const COMPARE_TOKENS = ["--c-1", "--c-2", "--c-3"];

function readTokens() {
  const s = getComputedStyle(document.documentElement);
  const get = (t) => s.getPropertyValue(t).trim();
  const domain = Object.fromEntries(DOMAINS.map((d) => [d.name, get(d.token)]));
  return {
    domain,
    compare: COMPARE_TOKENS.map(get),
    patent: get("--src-patent"),
    paper: get("--src-paper"),
    ink: get("--ink"),
    ink2: get("--ink-2"),
    muted: get("--muted"),
    grid: get("--grid"),
    surface: get("--surface"),
    accent: get("--accent"),
  };
}

// Resolved colour values for charts (SVG/canvas need real colours); updates on light/dark change.
export function useColors() {
  const [colors, setColors] = useState(readTokens);
  useEffect(() => {
    const mq = window.matchMedia("(prefers-color-scheme: dark)");
    const update = () => setColors(readTokens());
    mq.addEventListener("change", update);
    return () => mq.removeEventListener("change", update);
  }, []);
  return colors;
}

export const pct = (x, digits = 0) => `${x >= 0 ? "+" : ""}${(x * 100).toFixed(digits)}%`;
export const fmt = (n) => Number(n).toLocaleString();
