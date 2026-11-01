# %% [markdown]
# # 01 · Data exploration
#
# The cleaned dataset from Phase 2 (`data/processed/documents.parquet`): class balance,
# documents per year, top institutions and abstract lengths. Figures are saved to `reports/figures/`.

# %%
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid", context="notebook")
FIG = Path("reports/figures")
FIG.mkdir(parents=True, exist_ok=True)

df = pd.read_parquet("data/processed/documents.parquet")
splits = pd.read_parquet("data/processed/splits.parquet")
summary = json.loads(Path("reports/cleaning_summary.json").read_text())
print(f"{len(df):,} documents, {df['institution'].nunique():,} institutions")
df.head(3)[["doc_id", "source", "title", "institution_name", "country", "year", "domain"]]

# %% [markdown]
# ## Cleaning steps
# Records loaded and removed at each step (from `reports/cleaning_summary.json`).

# %%
pd.Series({k: v for k, v in summary.items() if k.startswith("dropped")} | {"loaded_papers": summary["loaded"]["paper"], "loaded_patents": summary["loaded"]["patent"]})

# %% [markdown]
# ## Class balance
# Each domain holds 6,000 documents (20%), well under the 30% limit. Robotics and Cybersecurity
# have fewer university patents, so papers fill the gap.

# %%
counts = df.groupby(["domain", "source"]).size().unstack()
ax = counts.plot.barh(stacked=True, figsize=(8, 4), color=["#4C72B0", "#DD8452"])
ax.set_xlabel("documents"); ax.set_ylabel(""); ax.set_title("Documents per domain and source")
plt.tight_layout(); plt.savefig(FIG / "class_balance.png", dpi=150); plt.show()
counts.assign(total=counts.sum(axis=1), share=lambda t: (t.total / t.total.sum()).round(3))

# %% [markdown]
# ## Documents per year

# %%
per_year = df.groupby(["year", "domain"]).size().unstack()
ax = per_year.plot(marker="o", figsize=(8, 4))
ax.set_ylabel("documents"); ax.set_title("Documents per year by domain")
plt.tight_layout(); plt.savefig(FIG / "documents_per_year.png", dpi=150); plt.show()
per_year

# %% [markdown]
# ## Top institutions and countries

# %%
top = df.groupby("institution_name").size().nlargest(15).sort_values()
ax = top.plot.barh(figsize=(8, 5), color="#55A868")
ax.set_xlabel("documents"); ax.set_ylabel(""); ax.set_title("Top 15 institutions")
plt.tight_layout(); plt.savefig(FIG / "top_institutions.png", dpi=150); plt.show()
both = df.groupby("institution")["source"].nunique()
print(f"Institutions with both patents and papers: {(both == 2).sum():,} of {len(both):,}")
df["country"].value_counts().head(10)

# %% [markdown]
# ## Abstract lengths

# %%
df["abstract_words"] = df["abstract"].str.split().str.len()
ax = sns.histplot(data=df, x="abstract_words", hue="source", bins=60, binrange=(30, 450), element="step", stat="density", common_norm=False)
ax.set_title("Abstract length (words)")
plt.tight_layout(); plt.savefig(FIG / "abstract_lengths.png", dpi=150); plt.show()
df.groupby("source")["abstract_words"].describe().round(1)

# %% [markdown]
# ## Splits
# Stratified 70/15/15 by domain with seed 42.

# %%
splits.merge(df[["doc_id", "domain"]], on="doc_id").groupby(["split", "domain"]).size().unstack()
