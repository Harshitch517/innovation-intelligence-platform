# %% [markdown]
# # 04 · Innovation clustering
#
# BERTopic (UMAP + HDBSCAN) on the MiniLM embeddings, across all documents (`G-*` clusters) and
# within each domain (`AI-*`, `HC-*`, `RB-*`, `CS-*`, `RE-*` sub-areas). Produced by
# `python -m src.pipeline.cluster`.

# %%
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from umap import UMAP

sns.set_theme(style="whitegrid")
FIG = Path("reports/figures"); FIG.mkdir(parents=True, exist_ok=True)
ev = json.loads(Path("reports/clustering_eval.json").read_text())
clusters = pd.read_parquet("data/outputs/clusters.parquet")
assign = pd.read_parquet("data/outputs/doc_clusters.parquet")
docs = pd.read_parquet("data/processed/documents.parquet").merge(assign, on="doc_id")
vec = np.load("data/processed/embeddings_minilm.npy")

# %% [markdown]
# ## Metrics
# - **Silhouette** (cosine, original 384-d space, outliers excluded) is low but positive: research
#   topics overlap and form a continuum rather than well-separated blobs.
# - **NMI** against the 5 domain labels measures how much the clusters agree with the domains.
#   **ARI** is much lower because it penalises splitting one domain into many sub-areas, which is
#   exactly what we want from 100 clusters.

# %%
g = ev["global"]
pd.DataFrame(
    [{"scope": "global", **{k: g[k] for k in ("clusters", "outlier_share", "silhouette_cosine_excl_outliers")},
      "nmi": g["nmi_vs_domain_excl_outliers"], "ari": g["ari_vs_domain_excl_outliers"]}]
    + [{"scope": d, **v} for d, v in ev["domain"].items()]
)

# %% [markdown]
# ## Global clusters in 2-D
# A 2-D UMAP of the embeddings coloured by domain, with the ten largest global clusters labelled.

# %%
xy = UMAP(n_neighbors=15, min_dist=0.1, metric="cosine", random_state=42).fit_transform(vec)
docs["x"], docs["y"] = xy[:, 0], xy[:, 1]
fig, ax = plt.subplots(figsize=(10, 8))
sns.scatterplot(data=docs, x="x", y="y", hue="domain", s=3, linewidth=0, alpha=0.5, ax=ax)
glob = clusters[clusters["scope"] == "global"].nlargest(10, "size")
for _, c in glob.iterrows():
    pts = docs.loc[docs["global_cluster"] == c["cluster_id"], ["x", "y"]].median()
    ax.text(pts["x"], pts["y"], ", ".join(list(c["keywords"])[:2]), fontsize=8, weight="bold",
            bbox=dict(facecolor="white", alpha=0.7, lw=0))
ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel(""); ax.set_ylabel("")
ax.legend(markerscale=4, loc="lower right"); ax.set_title("Documents in 2-D (UMAP), largest global clusters labelled")
plt.tight_layout(); plt.savefig(FIG / "clusters_umap.png", dpi=150); plt.show()

# %% [markdown]
# ## Cross-domain clusters
# Global clusters that mix domains show where technologies meet, such as AI methods applied to
# medical images.

# %%
mix = docs.groupby("global_cluster")["domain"].agg(lambda s: s.value_counts(normalize=True).iloc[0]).rename("top_domain_share")
glob_all = clusters[clusters["scope"] == "global"].set_index("cluster_id").join(mix)
glob_all["keywords"] = glob_all["keywords"].map(lambda k: ", ".join(list(k)[:6]))
glob_all.sort_values("top_domain_share")[["size", "top_domain_share", "keywords"]].head(10)

# %% [markdown]
# ## Largest sub-areas per domain

# %%
sub = clusters[clusters["scope"] == "domain"].copy()
sub["keywords"] = sub["keywords"].map(lambda k: ", ".join(list(k)[:6]))
sub.sort_values("size", ascending=False).groupby("domain").head(5)[["domain", "cluster_id", "size", "keywords"]]
