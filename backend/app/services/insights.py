"""A brief, local comparison of two documents: no LLM call, so it costs no API credits.

- key terms (words and two-word phrases) found in both documents, and in only one of them;
- the closest pair(s) of sentences, matched with the same MiniLM embeddings as search;
- a one-paragraph summary built from those facts.
"""
from __future__ import annotations

import re
from collections import Counter

import numpy as np

from app.services.embedding import embed_many

STOPWORDS = set("""
a about above after again against all also although am among an and any are as at be because been before being below
between both but by can could did do does doing down during each either else enable enables etc even ever every few for
from further had has have having here how however if in into is it its itself just less made make makes many may might
more most much must near no nor not now of off on once only onto or other others our out over own per rather same several
shall should since so some such than that the their them then there these they this those through thus to too under
until up upon us use used uses using very via was we were what when where whether which while who whom whose why will
with within without would yet you your new well high low large small different various based including include includes
included provide provides provided providing allow allows allowing within also one two three first second third further
""".split())
# Boilerplate common to patents and papers that says nothing about the technology.
BOILERPLATE = set("""
method methods system systems apparatus device devices invention present disclosed disclosure embodiment embodiments
comprising comprises comprise wherein thereof therein herein plurality said configured configuration according aspect
aspects example examples paper study studies propose proposed proposes approach approaches result results show shows
shown demonstrate demonstrates demonstrated performance experiment experiments experimental work article research
technique techniques framework novel related problem problems process processes step steps unit units module modules
application applications challenge challenges trend trends area areas field fields recent recently state art current
review overview survey highlight highlights important significant effective efficient improved improve improves
""".split())
STOP = STOPWORDS | BOILERPLATE
TOKEN_RE = re.compile(r"[a-z][a-z0-9\-]{2,}")
WORD_OR_PUNCT_RE = re.compile(r"[a-z0-9][a-z0-9\-]*|[^\sa-z0-9]")
SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")
MAX_SENTENCES = 12


def _norm(word: str) -> str:
    """Light plural folding so 'sensors' and 'sensor' count as one term."""
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 4 and word.endswith("s") and not word.endswith(("ss", "us", "is")):
        return word[:-1]
    return word


def terms(text: str) -> Counter:
    """Counts of content words and two-word phrases (phrases weighted double)."""
    counts: Counter = Counter()
    prev = None
    # Every word and punctuation mark is a token, so a phrase never skips a short word
    # ("fibrillation at home") or crosses a sentence or clause boundary.
    for raw in WORD_OR_PUNCT_RE.findall(text.lower()):
        t = _norm(raw)
        if not TOKEN_RE.fullmatch(raw) or t in STOP or t.isdigit():
            prev = None
            continue
        counts[t] += 1
        if prev:
            counts[f"{prev} {t}"] += 2
        prev = t
    return counts


def _top(counter: Counter, keys: set[str], n: int) -> list[str]:
    """Highest-weighted terms, skipping single words already covered by a chosen phrase."""
    ranked = sorted(keys, key=lambda k: (-counter[k], k))
    chosen: list[str] = []
    for k in ranked:
        if " " not in k and any(k in c.split() for c in chosen):
            continue
        if counter[k] < 2 and " " not in k:  # single words must appear at least twice
            continue
        chosen.append(k)
        if len(chosen) == n:
            break
    return chosen


def sentences(text: str) -> list[str]:
    parts = [s.strip() for s in SENT_RE.split(text) if len(s.split()) >= 6]
    return parts[:MAX_SENTENCES]


def compare(a: dict, b: dict, similarity: float) -> dict:
    """Brief comparison of two documents given as {"title", "abstract"} dicts."""
    ta, tb = terms(f"{a['title']}. {a['abstract']}"), terms(f"{b['title']}. {b['abstract']}")
    both = Counter({k: ta[k] + tb[k] for k in ta.keys() & tb.keys()})
    shared = _top(both, set(both), 8)
    # A term is distinctive only if it contains a word the other document never uses.
    words_a = {w for k in ta for w in k.split()}
    words_b = {w for k in tb for w in k.split()}
    only_a = _top(ta, {k for k in ta if any(w not in words_b for w in k.split())}, 5)
    only_b = _top(tb, {k for k in tb if any(w not in words_a for w in k.split())}, 5)
    vocab_a, vocab_b = {k for k in ta if " " not in k}, {k for k in tb if " " not in k}
    overlap = len(vocab_a & vocab_b) / max(1, len(vocab_a | vocab_b))

    passages = []
    sa, sb = sentences(a["abstract"]), sentences(b["abstract"])
    if sa and sb:
        va, vb = embed_many(sa), embed_many(sb)
        sim = va @ vb.T
        used_a, used_b = set(), set()
        for idx in np.argsort(-sim, axis=None):
            i, j = divmod(int(idx), sim.shape[1])
            if i in used_a or j in used_b:
                continue
            passages.append({"a": sa[i], "b": sb[j], "score": round(float(sim[i, j]), 4)})
            used_a.add(i)
            used_b.add(j)
            if len(passages) == 2:
                break

    pct = round(max(0.0, similarity) * 100)
    if shared:
        focus = ", ".join(shared[:3])
        summary = f"Both documents deal with {focus}."
    else:
        summary = "The two documents share few specific technical terms."
    if only_a or only_b:
        parts = []
        if only_a:
            parts.append(f"A additionally covers {', '.join(only_a[:3])}")
        if only_b:
            parts.append(f"B focuses on {', '.join(only_b[:3])}")
        summary += " " + "; ".join(parts) + "."
    level = "very closely" if pct >= 75 else "closely" if pct >= 60 else "partly" if pct >= 45 else "only loosely"
    summary += f" Overall their meanings match {level} ({pct}%), with {round(overlap * 100)}% of their vocabulary in common."
    if passages:
        summary += f" The closest passages match at {round(passages[0]['score'] * 100)}%."
    return {"summary": summary, "shared_terms": shared, "only_a": only_a, "only_b": only_b,
            "term_overlap": round(overlap, 4), "passages": passages}
