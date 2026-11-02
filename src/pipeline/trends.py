"""Phase 7a: innovation trends and emerging-technology detection.

Per sub-area (domain cluster) and year we count documents, then compute
  recent_growth  (n2024 + n2025 + 1) / (n2022 + n2023 + 1) - 1   growth of the last two years
  cagr           ((n2025 + 1) / (n2020 + 1)) ** (1 / 5) - 1        compound annual growth 2020-2025
and rank clusters with at least `min_size` documents by recent growth (the emerging list).

Back-test (PLAN.md Section 7): using 2020-2023 only, flag the top `top_n` clusters by
(n2022 + n2023 + 1) / (n2020 + n2021 + 1); a flag is a hit if the cluster's yearly average in
2024-2025 is higher than in 2022-2023. The hit rate is compared with the base rate of all clusters.

Counts come from seeded random samples drawn across all years (Phase 1), so year-to-year ratios
within a source follow the real publication and grant volumes.

Outputs: data/outputs/trends.parquet (cluster_id, year, count), data/outputs/emerging.parquet,
         data/outputs/domain_trends.parquet, reports/trend_backtest.json

Usage:
    python -m src.pipeline.trends
"""
from __future__ import annotations

import json

import pandas as pd

from src.config import load_config, path

MIN_SIZE = 40
TOP_N = 20


def yearly_counts(docs: pd.DataFrame, years: range) -> pd.DataFrame:
    """Cluster x year document counts (missing years filled with 0)."""
    table = docs.groupby(["domain_cluster", "year"]).size().unstack(fill_value=0)
    return table.reindex(columns=list(years), fill_value=0)


def growth_table(counts: pd.DataFrame) -> pd.DataFrame:
    """Recent growth, CAGR and size for every cluster."""
    c = counts
    out = pd.DataFrame(index=c.index)
    out["size"] = c.sum(axis=1)
    out["recent_growth"] = (c[2024] + c[2025] + 1) / (c[2022] + c[2023] + 1) - 1
    out["cagr"] = ((c[2025] + 1) / (c[2020] + 1)) ** (1 / 5) - 1
    return out.round(4)


def backtest(counts: pd.DataFrame, min_size: int, top_n: int) -> dict:
    """Flag clusters from 2020-2023 data only and check whether they grew in 2024-2025."""
    c = counts
    early_size = c[[2020, 2021, 2022, 2023]].sum(axis=1)
    eligible = c[early_size >= min_size * 4 / 6]  # same size rule, scaled to four years
    early_score = (eligible[2022] + eligible[2023] + 1) / (eligible[2020] + eligible[2021] + 1)
    flagged = early_score.nlargest(top_n).index
    grew = (eligible[2024] + eligible[2025]) / 2 > (eligible[2022] + eligible[2023]) / 2
    later_ratio = ((eligible[2024] + eligible[2025] + 1) / (eligible[2022] + eligible[2023] + 1))
    return {
        "rule": f"flag top {top_n} clusters by (n2022+n2023+1)/(n2020+n2021+1) using 2020-2023 only; "
                "hit = yearly average 2024-25 above 2022-23",
        "eligible_clusters": int(len(eligible)),
        "flagged": int(len(flagged)),
        "hit_rate": round(float(grew[flagged].mean()), 4),
        "base_rate_all_eligible": round(float(grew.mean()), 4),
        "base_rate_not_flagged": round(float(grew.drop(flagged).mean()), 4),
        "mean_later_growth_flagged": round(float(later_ratio[flagged].mean() - 1), 4),
        "mean_later_growth_not_flagged": round(float(later_ratio.drop(flagged).mean() - 1), 4),
        "flagged_clusters": [str(x) for x in flagged],
    }


def main() -> None:
    """Compute trends, the emerging list and the back-test, and save them."""
    cfg = load_config()
    years = range(cfg["years"]["start"], cfg["years"]["end"] + 1)
    docs = pd.read_parquet(path("documents")).merge(pd.read_parquet(path("outputs") / "doc_clusters.parquet"), on="doc_id")
    clusters = pd.read_parquet(path("outputs") / "clusters.parquet").set_index("cluster_id")

    counts = yearly_counts(docs, years)
    growth = growth_table(counts)
    growth["domain"] = clusters.loc[growth.index, "domain"]
    growth["keywords"] = clusters.loc[growth.index, "keywords"].map(lambda k: list(k)[:8])
    emerging = growth[growth["size"] >= MIN_SIZE].sort_values("recent_growth", ascending=False)
    emerging["rank"] = range(1, len(emerging) + 1)

    out = path("outputs")
    long = counts.stack().rename("count").reset_index().rename(columns={"domain_cluster": "cluster_id"})
    long.to_parquet(out / "trends.parquet", index=False)
    emerging.reset_index(names="cluster_id").to_parquet(out / "emerging.parquet", index=False)
    domain_year = docs.groupby(["domain", "year", "source"]).size().rename("count").reset_index()
    domain_year.to_parquet(out / "domain_trends.parquet", index=False)

    bt = backtest(counts, MIN_SIZE, TOP_N)
    bt["min_size"] = MIN_SIZE
    bt["top_emerging_now"] = [
        {"cluster_id": i, "domain": r["domain"], "recent_growth": r["recent_growth"], "cagr": r["cagr"],
         "size": int(r["size"]), "keywords": r["keywords"][:5]}
        for i, r in emerging.head(10).iterrows()
    ]
    (path("reports") / "trend_backtest.json").write_text(json.dumps(bt, indent=2), encoding="utf-8")
    print(f"Back-test: hit rate {bt['hit_rate']} vs base rate {bt['base_rate_not_flagged']} (not flagged), "
          f"{bt['flagged']} flagged of {bt['eligible_clusters']}")
    print("Top emerging now:")
    for e in bt["top_emerging_now"]:
        print(f"  {e['cluster_id']:<6} {e['domain']:<24} growth {e['recent_growth']:+.0%}  {', '.join(e['keywords'])}")


if __name__ == "__main__":
    main()
