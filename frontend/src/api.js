// Thin client for the FastAPI backend. The base URL comes from VITE_API_URL (Vercel) and
// defaults to the local backend.
export const API_URL = (import.meta.env.VITE_API_URL || "http://localhost:8000").replace(/\/$/, "");

async function request(path, options = {}) {
  const isForm = options.body instanceof FormData;
  const res = await fetch(`${API_URL}${path}`, {
    headers: isForm ? {} : { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch {
      /* keep statusText */
    }
    throw new Error(`${res.status}: ${detail}`);
  }
  return res.json();
}

function query(params) {
  const q = new URLSearchParams();
  Object.entries(params).forEach(([k, v]) => {
    if (v !== undefined && v !== null && v !== "") q.set(k, v);
  });
  const s = q.toString();
  return s ? `?${s}` : "";
}

export const api = {
  search: (params) => request(`/search${query(params)}`),
  document: (id) => request(`/documents/${encodeURIComponent(id)}`),
  similar: (id, params) => request(`/documents/${encodeURIComponent(id)}/similar${query(params)}`),
  compare: (a, b) => request(`/compare${query({ a, b })}`),
  explain: (a, b) => request(`/explain${query({ a, b })}`),
  classify: (title, abstract) => request("/classify", { method: "POST", body: JSON.stringify({ title, abstract }) }),
  analyze: (file, k = 10) => {
    const form = new FormData();
    form.append("file", file);
    return request(`/analyze?k=${k}`, { method: "POST", body: form });
  },
  compareUpload: (title, abstract, docId) =>
    request("/compare/upload", { method: "POST", body: JSON.stringify({ title, abstract, doc_id: docId }) }),
  explainUpload: (title, abstract, docId) =>
    request("/explain/upload", { method: "POST", body: JSON.stringify({ title, abstract, doc_id: docId }) }),
  overview: () => request("/stats/overview"),
  trends: (domain) => request(`/trends${query({ domain })}`),
  clusters: (params) => request(`/clusters${query(params)}`),
  cluster: (id) => request(`/clusters/${encodeURIComponent(id)}`),
  institutions: (params) => request(`/institutions${query(params)}`),
  institution: (name) => request(`/institutions/${encodeURIComponent(name)}`),
  network: (params) => request(`/institutions/network${query(params)}`),
};
