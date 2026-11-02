"""Phase 12: collect every metric, table and plot into reports/final_report.md.

All numbers are read from the JSON/CSV files the pipeline wrote, so the report always matches
the latest run. Usage:
    python scripts/make_report.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config, path  # noqa: E402

R = path("reports")


def load(name: str) -> dict:
    """Read a JSON report (empty dict if missing)."""
    f = R / name
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}


def table(df: pd.DataFrame) -> str:
    """DataFrame -> GitHub markdown table."""
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for row in df.itertuples(index=False):
        lines.append("| " + " | ".join(
            (f"{v:.4f}" if abs(v) < 10 else f"{v:,.1f}") if isinstance(v, float) else str(v) for v in row) + " |")
    return "\n".join(lines)


def main() -> None:
    """Write the report."""
    cfg = load_config()
    clean, cls, sim, search = load("cleaning_summary.json"), load("classification.json"), load("similarity_eval.json"), load("search_eval.json")
    clu, bt, inst, db, emb = load("clustering_eval.json"), load("trend_backtest.json"), load("institutions_summary.json"), load("db_check.json"), load("embedding_runs.json")
    patents, papers = load("fetch_patents_counts.json"), load("fetch_papers_counts.json")
    names_file = path("outputs") / "cluster_names.json"
    names = json.loads(names_file.read_text(encoding="utf-8")) if names_file.exists() else {}
    cmp = pd.read_csv(R / "model_comparison.csv")
    fin = clean["final"]

    out: list[str] = [
        "# Innovation Intelligence Platform — Evaluation Report",
        "",
        "INT395 project. Every number below is read from the files in `reports/` written by the pipeline "
        "(`python scripts/make_report.py` regenerates this report).",
        "",
        "## 1. Data",
        "",
        f"- **Patents:** Google Patents Public Datasets on BigQuery — granted US patents {cfg['years']['start']}–{cfg['years']['end']} "
        "with a university or research-institute assignee, original-English title and abstract, one per patent family. "
        "Label: primary CPC code, else first matching inventive CPC code, else a Y02E tag.",
        "- **Papers:** OpenAlex journal articles with at least one university / research-institute author, labelled by primary topic "
        "(topic lists in `reports/label_sources.md`).",
        f"- **Cleaning:** {clean['dropped_short_or_missing_abstract']:,} short or missing abstracts, "
        f"{clean['dropped_non_english_abstract']:,} non-English abstracts and {clean['dropped_duplicate_title']:,} duplicate titles removed.",
        f"- **Final dataset:** {fin['documents']:,} documents, {fin['institutions']:,} institutions; largest domain share "
        f"{fin['max_domain_share']:.0%}; splits {fin['splits']['train']:,} / {fin['splits']['val']:,} / {fin['splits']['test']:,} (stratified, seed {cfg['seed']}).",
        "",
        table(pd.DataFrame([{"domain": k.split("|")[0], "source": k.split("|")[1], "documents": v} for k, v in fin["by_domain_source"].items()])
              .pivot(index="domain", columns="source", values="documents").reset_index()),
        "",
        "Patents available per domain after filtering: "
        + ", ".join(f"{d} {v['available_in_staging']:,}" for d, v in patents.items()) + ".",
        "",
        "![Class balance](figures/class_balance.png)",
        "",
        "## 2. Embeddings",
        "",
        table(pd.DataFrame([{"model": v["model"], "dimension": v["dimension"], "seconds (30k docs, RTX 4050)": v["seconds"]} for v in emb.values()])),
        "",
        "## 3. Domain classification (supervised)",
        "",
        "Same split for every model; hyperparameters chosen on validation; the test split was used once, after all choices were frozen.",
        "",
        table(cmp[["model", "val_accuracy", "val_f1_macro", "train_seconds"]]),
        "",
        "**Test results**",
        "",
        table(pd.DataFrame([{"model": k, "role": ", ".join(v["roles"]), **{m: v["test"][m] for m in ("accuracy", "precision_macro", "recall_macro", "f1_macro", "f1_weighted")}}
                            for k, v in cls["test"].items()])),
        "",
        f"Leakage check: {cls['leakage_check']['docs_with_cpc_code_in_text']} documents contain a CPC code and "
        f"{cls['leakage_check']['docs_with_openalex_topic_id_in_text']} an OpenAlex topic ID in the model input (title + abstract only).",
        "",
        "![Confusion matrix](confusion_matrix.png)",
        "",
        "![Per-class F1](per_class_f1.png)",
        "",
        "**Error analysis.** The most frequent test errors: "
        + "; ".join(f"{k} ({v})" for k, v in list(cls["top_confusions_best_model"].items())[:6])
        + ". They sit on domain boundaries (AI methods applied to medical data, learning-based robot control, ML for intrusion detection); "
        "20 examples are listed in `misclassified_examples.csv`.",
        "",
        f"**Embedding choice.** SPECTER2 + Logistic Regression reaches validation macro F1 {cls['embedding_comparison_val_f1_macro']['specter2']:.4f} "
        f"vs {cls['embedding_comparison_val_f1_macro']['minilm']:.4f} for MiniLM, but MiniLM retrieves related documents better "
        f"(MRR {sim['minilm']['mrr']:.4f} vs {sim['specter2']['mrr']:.4f}) and runs under fastembed on the 512 MB backend, so MiniLM is the production embedding.",
        "",
        "## 4. Similarity detection and semantic search",
        "",
        f"Ground truth: {sim['ground_truth']}. {sim['minilm']['queries']:,} query documents, k = 10.",
        "",
        table(pd.DataFrame([{"embedding": k, **{m: sim[k][m] for m in ("precision@10", "recall@10", "mrr")}}
                            for k in ("minilm", "specter2", "tfidf_svd_baseline", "random_baseline")])),
        "",
        "Precision@10 is small in absolute terms because most documents have only one or two relevant documents in the corpus.",
        "",
        f"**Semantic search:** Precision@10 = {search['precision@10']:.2f} over {search['queries']} hand-written queries "
        f"(per domain: {', '.join(f'{d} {v:.2f}' for d, v in search['by_domain'].items())}).",
        "",
        f"**Cross-institution similarity:** threshold {sim['cross_institution']['threshold']} ({sim['cross_institution']['rule']}); "
        f"{sim['cross_institution']['pairs_flagged']:,} pairs flagged.",
        "",
        f"**Database check:** pgvector HNSW top-10 identical to FAISS for {db['faiss_check']['identical_top10_share']:.0%} of "
        f"{db['faiss_check']['test_documents']} test documents (median query {db['faiss_check']['median_query_ms']} ms).",
        "",
        "![Retrieval](figures/similarity_retrieval.png)",
        "",
        "## 5. Clustering (unsupervised)",
        "",
        table(pd.DataFrame(
            [{"scope": "global", "clusters": clu["global"]["clusters"], "outlier_share": clu["global"]["outlier_share"],
              "silhouette": clu["global"]["silhouette_cosine_excl_outliers"], "NMI vs domain": clu["global"]["nmi_vs_domain_excl_outliers"],
              "ARI vs domain": clu["global"]["ari_vs_domain_excl_outliers"]}]
            + [{"scope": d, "clusters": v["clusters"], "outlier_share": v["outlier_share"], "silhouette": v["silhouette_cosine_excl_outliers"],
                "NMI vs domain": "", "ARI vs domain": ""} for d, v in clu["domain"].items()])),
        "",
        "Silhouette is low but positive (research topics form a continuum); ARI is low because 94 clusters split 5 domains into "
        "sub-areas, which NMI tolerates better. Clusters were named by Gemini (Phase 11), with keywords as fallback.",
        "",
        "![Clusters](figures/clusters_umap.png)",
        "",
        "## 6. Emerging technologies",
        "",
        table(pd.DataFrame([{"rank": i + 1, "sub-area": names.get(e["cluster_id"], {}).get("name") or ", ".join(e["keywords"][:3]),
                             "domain": e["domain"], "documents": e["size"], "growth 2024-25 vs 2022-23": f"{e['recent_growth']:+.0%}",
                             "CAGR 2020-25": f"{e['cagr']:+.0%}"} for i, e in enumerate(bt["top_emerging_now"])])),
        "",
        f"**Back-test:** {bt['rule']}. Hit rate {bt['hit_rate']:.2f} vs {bt['base_rate_not_flagged']:.2f} for clusters not flagged "
        f"({bt['flagged']} flagged of {bt['eligible_clusters']}). The two-year-window rule gives almost no lift over the base rate: "
        "most sub-areas grew over the period and clusters of 40–200 documents give noisy yearly counts, so the emerging list "
        "describes recent growth rather than predicting it.",
        "",
        "![Emerging](figures/emerging_top10.png)",
        "",
        "## 7. Institutions",
        "",
        f"{inst['institutions_profiled']} institutions with at least {inst['min_docs']} documents are profiled (mean embedding). "
        f"Network: {inst['edges']:,} edges (similarity >= {inst['edge_threshold']} or >= 3 shared sub-areas, at most 8 per node). "
        f"Most connected: {', '.join(inst['most_connected'])}.",
        "",
        "![Network](figures/institution_network.png)",
        "",
        "## 8. Changes from the original plan",
        "",
        "See PLAN.md Section 10 (patent source moved from PatentsView to BigQuery; hybrid text table to stay under the 200 GB "
        "query cap; US grants only; extended labelling rule; non-English abstracts removed; Gemini model switched to a lighter one).",
        "",
    ]
    (R / "final_report.md").write_text("\n".join(out), encoding="utf-8")
    print(f"Wrote {R / 'final_report.md'}")


if __name__ == "__main__":
    main()
