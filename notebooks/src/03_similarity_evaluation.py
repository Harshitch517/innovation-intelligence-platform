# %% [markdown]
# # 03 · Similarity and semantic search evaluation
#
# Ground truth for similarity: two documents are relevant to each other if they share a primary
# CPC subgroup (patents) or one cites the other. Metrics are averaged over every document that
# has at least one relevant document. Produced by `python -m src.pipeline.similarity`.

# %%
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import load_config
from src.pipeline.similarity import SimilarityIndex, choose_threshold, neighbour_pairs

sns.set_theme(style="whitegrid")
FIG = Path("reports/figures"); FIG.mkdir(parents=True, exist_ok=True)
cfg = load_config()
sim = json.loads(Path("reports/similarity_eval.json").read_text())
search = json.loads(Path("reports/search_eval.json").read_text())
docs = pd.read_parquet("data/processed/documents.parquet")
vec = np.load("data/processed/embeddings_minilm.npy")

# %% [markdown]
# ## Retrieval quality by embedding
# MiniLM ranks relevant documents highest (MRR and Recall@10), ahead of SPECTER2 and a TF-IDF + SVD
# baseline; random vectors give a floor of about zero. Precision@10 is small in absolute terms
# because most documents have only one or two relevant documents in the corpus, so even a
# perfect ranking cannot fill ten slots.

# %%
names = {"minilm": "MiniLM", "specter2": "SPECTER2", "tfidf_svd_baseline": "TF-IDF + SVD", "random_baseline": "Random"}
table = pd.DataFrame({names[k]: sim[k] for k in names}).T[["precision@10", "recall@10", "mrr"]]
ax = table.plot.bar(figsize=(8, 4), rot=0)
ax.set_title(f"Retrieval on {sim['minilm']['queries']:,} query documents (k = 10)")
plt.tight_layout(); plt.savefig(FIG / "similarity_retrieval.png", dpi=150); plt.show()
table

# %% [markdown]
# ## Choosing the cross-institution similarity threshold
# For every document we take its 10 nearest neighbours from *other* institutions. The threshold
# is the lowest cosine similarity at which at least 95% of those pairs share a domain.

# %%
index = SimilarityIndex(docs, vec, cfg["models"]["embedding"])
pairs = neighbour_pairs(index)
threshold = choose_threshold(pairs)
fig, ax = plt.subplots(figsize=(8, 4))
sns.histplot(data=pairs, x="score", hue="same_domain", bins=80, element="step", ax=ax)
ax.axvline(threshold, color="black", ls="--"); ax.text(threshold + 0.005, ax.get_ylim()[1] * 0.9, f"threshold {threshold}")
ax.set_title("Cross-institution neighbour pairs by cosine similarity")
plt.tight_layout(); plt.savefig(FIG / "similarity_threshold.png", dpi=150); plt.show()
bins = pd.cut(pairs["score"], np.arange(0.3, 1.01, 0.05))
pairs.groupby(bins, observed=True)["same_domain"].agg(["mean", "size"]).round(3)

# %% [markdown]
# ## Example: similar innovations from other institutions

# %%
example = docs[(docs["source"] == "patent") & (docs["domain"] == "Healthcare")].iloc[0]
print(example["title"], "—", example["institution_name"])
index.similar(example["doc_id"], 5)[["title", "institution_name", "source", "domain", "score", "other_institution"]]

# %% [markdown]
# ## Semantic search: 20 hand-written queries
# Precision@10 counts a result as correct when its domain matches the query's intended domain.

# %%
print("Overall Precision@10:", search["precision@10"])
pd.DataFrame(search["per_query"])[["domain", "query", "precision@10"]]

# %%
index.search("deep learning for retinal disease detection", 5, domain="Healthcare")[["title", "institution_name", "year", "score"]]
