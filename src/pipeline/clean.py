"""Phase 2: clean, label, balance and split the raw patents and papers.

Input:  data/raw/papers/<domain>/*.json (OpenAlex pages) and data/raw/patents/*.parquet
Output: data/processed/documents.parquet  (unified schema, PLAN.md Section 5)
        data/processed/splits.parquet     (doc_id -> train / val / test)
        reports/cleaning_summary.json     (what was dropped at each step, final counts)

Steps
1. Papers: rebuild the abstract from OpenAlex's inverted index, take the first university /
   research-institute affiliation as the institution, label by primary topic.
2. Both sources: strip markup, drop abstracts shorter than min_abstract_words, drop duplicate
   records (same ID, then same normalised title across all sources).
3. Normalise institution names so "UNIV CALIFORNIA" (patents) and "University of California"
   (papers) become the same key.
4. Balance: per domain at most max_patents_per_domain patents, then papers up to
   target_per_domain documents (seeded random sampling).
5. Stratified 70/15/15 train/validation/test split by domain (seed from config.yaml).

Usage:
    python -m src.pipeline.clean
"""
from __future__ import annotations

import html
import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import load_config, path

COLUMNS = [
    "doc_id", "source", "title", "abstract", "text", "institution", "institution_name",
    "institution_raw", "country", "year", "domain", "raw_codes", "cited_ids",
]
TAG_RE = re.compile(r"<[^>]+>")
LEADING_ABSTRACT_RE = re.compile(r"^\s*(abstract|summary)\s*[:.\-–—]?\s*", re.IGNORECASE)


# ---------------------------------------------------------------- text helpers

def rebuild_abstract(inverted_index: dict[str, list[int]] | None) -> str:
    """Turn OpenAlex's {word: [positions]} inverted index back into plain text."""
    if not inverted_index:
        return ""
    positions = [(pos, word) for word, poss in inverted_index.items() for pos in poss]
    return " ".join(word for _, word in sorted(positions))


def clean_text(text: str | None) -> str:
    """Remove HTML/JATS tags and entities, a leading "Abstract:" label and extra whitespace."""
    if not text:
        return ""
    text = html.unescape(TAG_RE.sub(" ", text))
    text = LEADING_ABSTRACT_RE.sub("", text)
    return re.sub(r"\s+", " ", text).strip()


def normalise_title(title: str) -> str:
    """Lowercase title with only letters and digits, used to find duplicates."""
    return re.sub(r"[^a-z0-9]+", " ", title.lower()).strip()


def _name_key(name: str, c: dict) -> str:
    """Order-independent key for an institution name (no alias lookup)."""
    words = re.sub(r"[^a-z0-9]+", " ", name.lower()).split()
    words = [c["abbreviations"].get(w, w) for w in words]
    drop = set(c["stopwords"]) | set(c["noise_words"])
    if "cooperation" in words:
        drop |= set(c["cooperation_words"])
    kept = [w for w in words if w not in drop]
    # Word order differs between sources ("UNIV TSINGHUA" vs "Tsinghua University"), so sort.
    return " ".join(sorted(dict.fromkeys(kept or words)))


def normalise_institution(name: str | None, cfg: dict) -> str:
    """Normalised institution key: lowercase, no punctuation, abbreviations expanded,
    stopwords and legal-entity words removed, words sorted, known aliases mapped.

    "UNIV TSINGHUA" and "Tsinghua University" both become "tsinghua university";
    "SEOUL NAT UNIV R&DB FOUND" becomes "national seoul university".
    """
    if not name:
        return ""
    c = cfg["cleaning"]
    aliases = {_name_key(k, c): _name_key(v, c) for k, v in c["aliases"].items()}
    key = _name_key(name, c)
    return aliases.get(key, key)


def display_names(df: pd.DataFrame, cfg: dict) -> pd.Series:
    """Readable name per institution key: the most common paper spelling (proper case),
    else the most common patent spelling with abbreviations expanded, in title case
    ("IND TECH RES INST" -> "Industrial Technology Research Institute")."""
    abbreviations = cfg["cleaning"]["abbreviations"]

    def pick(group: pd.DataFrame) -> str:
        papers = group.loc[group["source"] == "paper", "institution_raw"]
        if len(papers):
            return papers.value_counts().index[0]
        raw = group["institution_raw"].value_counts().index[0]
        return " ".join(abbreviations.get(w.lower(), w.lower()).title() for w in raw.split())

    return df.groupby("institution")[["source", "institution_raw"]].apply(pick)


def word_count(text: str) -> int:
    """Number of whitespace-separated words."""
    return len(text.split())


# ---------------------------------------------------------------- loaders

def topic_to_domain(cfg: dict) -> dict[str, str]:
    """Map every configured OpenAlex topic ID (e.g. "T10320") to its domain."""
    return {t: d for d, spec in cfg["domains"].items() for t in spec["openalex_topics"]}


def paper_record(work: dict, topic_domain: dict[str, str]) -> dict | None:
    """Convert one OpenAlex work into the unified schema (None if it has no usable label)."""
    primary = (work.get("primary_topic") or {}).get("id", "").rsplit("/", 1)[-1]
    domain = topic_domain.get(primary)
    if domain is None:
        return None
    institution, country = "", ""
    for authorship in work.get("authorships", []):
        for inst in authorship.get("institutions", []):
            if inst.get("type") in ("education", "facility"):
                institution, country = inst.get("display_name", ""), inst.get("country_code") or ""
                break
        if institution:
            break
    topics = [t["id"].rsplit("/", 1)[-1] for t in work.get("topics", [])]
    work_id = work["id"].rsplit("/", 1)[-1]
    return {
        "doc_id": f"PAP-{work_id}",
        "source": "paper",
        "title": clean_text(work.get("title")),
        "abstract": clean_text(rebuild_abstract(work.get("abstract_inverted_index"))),
        "institution_raw": institution,
        "country": country,
        "year": int(work["publication_year"]),
        "domain": domain,
        "raw_codes": [primary] + [t for t in topics if t != primary],
        "cited_ids": [f"PAP-{w.rsplit('/', 1)[-1]}" for w in work.get("referenced_works", [])],
    }


def load_papers(raw_dir: Path, cfg: dict) -> pd.DataFrame:
    """Read every saved OpenAlex page and return one row per unique work."""
    topic_domain = topic_to_domain(cfg)
    rows: dict[str, dict] = {}
    for file in sorted(raw_dir.glob("*/*.json")):
        for work in json.loads(file.read_text(encoding="utf-8"))["results"]:
            rec = paper_record(work, topic_domain)
            if rec is not None:
                rows.setdefault(rec["doc_id"], rec)
    return pd.DataFrame(list(rows.values()))


def load_patents(raw_dir: Path) -> pd.DataFrame:
    """Read the per-domain patent parquet files written by fetch_patents.py."""
    frames = [pd.read_parquet(f) for f in sorted(raw_dir.glob("*.parquet"))]
    df = pd.concat(frames, ignore_index=True).drop_duplicates("doc_id")
    df = df.rename(columns={"institution": "institution_raw"})
    df["title"] = df["title"].map(clean_text)
    df["abstract"] = df["abstract"].map(clean_text)
    df["raw_codes"] = df["raw_codes"].map(list)
    df["cited_ids"] = df["cited_ids"].map(list)
    return df


# ---------------------------------------------------------------- main steps

def filter_and_dedupe(df: pd.DataFrame, min_words: int, log: dict) -> pd.DataFrame:
    """Drop short/missing abstracts and duplicate titles; record how many rows each step removed."""
    n0 = len(df)
    df = df[(df["title"] != "") & (df["abstract"].map(word_count) >= min_words)]
    log["dropped_short_or_missing_abstract"] = n0 - len(df)

    n1 = len(df)
    # Patents first so that a paper duplicating a patent title is the one dropped.
    df = df.assign(_t=df["title"].map(normalise_title)).sort_values(["source", "doc_id"], ascending=[False, True])
    df = df.drop_duplicates("_t").drop(columns="_t")
    log["dropped_duplicate_title"] = n1 - len(df)
    return df


def balance(df: pd.DataFrame, max_patents: int, target: int, seed: int) -> pd.DataFrame:
    """Per domain: up to `max_patents` patents, then papers to reach `target` documents."""
    parts = []
    for _, group in df.groupby("domain"):
        patents = group[group["source"] == "patent"]
        patents = patents.sample(n=min(len(patents), max_patents), random_state=seed)
        papers = group[group["source"] == "paper"]
        papers = papers.sample(n=min(len(papers), max(0, target - len(patents))), random_state=seed)
        parts += [patents, papers]
    return pd.concat(parts).sort_values("doc_id").reset_index(drop=True)


def make_splits(df: pd.DataFrame, fractions: dict, seed: int) -> pd.DataFrame:
    """Stratified train/val/test split by domain; returns doc_id + split."""
    train, rest = train_test_split(df, train_size=fractions["train"], stratify=df["domain"], random_state=seed)
    val_share = fractions["val"] / (fractions["val"] + fractions["test"])
    val, test = train_test_split(rest, train_size=val_share, stratify=rest["domain"], random_state=seed)
    return pd.concat(
        [train[["doc_id"]].assign(split="train"), val[["doc_id"]].assign(split="val"), test[["doc_id"]].assign(split="test")]
    ).sort_values("doc_id").reset_index(drop=True)


def main() -> None:
    """Run all cleaning steps and write documents.parquet, splits.parquet and a summary."""
    cfg = load_config()
    c = cfg["cleaning"]
    log: dict = {}

    papers = load_papers(path("raw_papers"), cfg)
    patents = load_patents(path("raw_patents"))
    log["loaded"] = {"paper": len(papers), "patent": len(patents)}
    print(f"Loaded {len(papers):,} papers and {len(patents):,} patents")

    df = pd.concat([patents, papers], ignore_index=True)
    df = df[(df["year"] >= cfg["years"]["start"]) & (df["year"] <= cfg["years"]["end"])]
    df = filter_and_dedupe(df, c["min_abstract_words"], log)

    df["institution"] = df["institution_raw"].map(lambda n: normalise_institution(n, cfg))
    log["dropped_no_institution"] = int((df["institution"] == "").sum())
    df = df[df["institution"] != ""]
    df["institution_name"] = df["institution"].map(display_names(df, cfg))
    log["available_after_cleaning"] = {
        f"{d}|{s}": int(n) for (d, s), n in df.groupby(["domain", "source"]).size().items()
    }

    df = balance(df, c["max_patents_per_domain"], cfg["target_per_domain"], cfg["seed"])
    df["text"] = df["title"] + ". " + df["abstract"]
    df = df[COLUMNS + [col for col in df.columns if col not in COLUMNS]]

    splits = make_splits(df, cfg["splits"], cfg["seed"])
    out_docs, out_proc = path("documents"), path("processed")
    df.to_parquet(out_docs, index=False)
    splits.to_parquet(out_proc / "splits.parquet", index=False)

    share = df["domain"].value_counts(normalize=True)
    log["final"] = {
        "documents": len(df),
        "by_domain_source": {f"{d}|{s}": int(n) for (d, s), n in df.groupby(["domain", "source"]).size().items()},
        "by_domain": df["domain"].value_counts().sort_index().to_dict(),
        "max_domain_share": round(float(share.max()), 4),
        "by_year": {int(k): int(v) for k, v in df["year"].value_counts().sort_index().items()},
        "splits": splits["split"].value_counts().to_dict(),
        "institutions": int(df["institution"].nunique()),
    }
    (path("reports") / "cleaning_summary.json").write_text(json.dumps(log, indent=2), encoding="utf-8")

    print(f"Wrote {len(df):,} documents to {out_docs}")
    print("By domain/source:", log["final"]["by_domain_source"])
    print(f"Largest domain share: {share.max():.1%} | splits: {log['final']['splits']}")
    print("Dropped:", {k: v for k, v in log.items() if k.startswith("dropped")})


if __name__ == "__main__":
    main()
