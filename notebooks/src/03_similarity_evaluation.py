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
from src.pipeline.similarity import SimilarityIndex

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
# ## Example: most similar documents

# %%
index = SimilarityIndex(docs, vec, cfg["models"]["embedding"])
example = docs[(docs["source"] == "patent") & (docs["domain"] == "Healthcare")].iloc[0]
print(example["title"])
index.similar(example["doc_id"], 5)[["title", "source", "domain", "score"]]

# %% [markdown]
# ## Semantic search: 20 hand-written queries
# Precision@10 counts a result as correct when its domain matches the query's intended domain.

# %%
print("Overall Precision@10:", search["precision@10"])
pd.DataFrame(search["per_query"])[["domain", "query", "precision@10"]]

# %%
index.search("deep learning for retinal disease detection", 5, domain="Healthcare")[["title", "institution_name", "year", "score"]]
