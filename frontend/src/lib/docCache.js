// In-memory cache of the most recently opened documents. Nothing goes to localStorage or disk,
// so closing the tab or site clears everything.
const MAX_ITEMS = 5;

class RecentCache {
  constructor(limit, onEvict) {
    this.limit = limit;
    this.onEvict = onEvict;
    this.items = new Map(); // Map keeps insertion order: the first key is the oldest
  }

  get(key) {
    if (!this.items.has(key)) return undefined;
    const value = this.items.get(key);
    this.items.delete(key);
    this.items.set(key, value); // reopened: now the newest
    return value;
  }

  set(key, value) {
    if (this.items.has(key)) this.onEvict?.(this.items.get(key));
    this.items.delete(key);
    this.items.set(key, value);
    while (this.items.size > this.limit) {
      const oldest = this.items.keys().next().value;
      this.onEvict?.(this.items.get(oldest));
      this.items.delete(oldest);
    }
  }

  get size() {
    return this.items.size;
  }
}

export const detailsCache = new RecentCache(MAX_ITEMS);
// PDFs are held as blob URLs; revoking one frees its memory.
export const pdfCache = new RecentCache(MAX_ITEMS, (url) => URL.revokeObjectURL(url));
export const PDF_CACHE_LIMIT = MAX_ITEMS;

// Download a PDF into a blob URL, reporting (received bytes, total bytes or null) as it arrives.
export async function downloadPdf(url, onProgress, signal) {
  const res = await fetch(url, { signal });
  if (!res.ok) throw new Error("PDF could not be loaded");
  const total = Number(res.headers.get("content-length")) || null;
  const reader = res.body.getReader();
  const parts = [];
  let received = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    parts.push(value);
    received += value.length;
    onProgress(received, total);
  }
  return URL.createObjectURL(new Blob(parts, { type: "application/pdf" }));
}
