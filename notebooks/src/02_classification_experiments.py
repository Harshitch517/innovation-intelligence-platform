# %% [markdown]
# # 02 · Domain classification experiments
#
# Four model families trained on the same stratified 70/15/15 split (seed 42), tuned on the
# validation split only. The test split was used once, after all choices were frozen, for the
# best model (fine-tuned SciBERT) and for the model the API serves (MiniLM embeddings + Logistic
# Regression, small enough for the free 512 MB backend). Produced by
# `python -m src.pipeline.train_classifier`.

# %%
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from IPython.display import Image, display

sns.set_theme(style="whitegrid")
rep = json.loads(Path("reports/classification.json").read_text())
cmp = pd.read_csv("reports/model_comparison.csv")

# %% [markdown]
# ## Model comparison (validation split)

# %%
cmp[["model", "val_accuracy", "val_f1_macro", "train_seconds"]]

# %%
ax = cmp.set_index("model")["val_f1_macro"].sort_values().plot.barh(figsize=(8, 3.5), xlim=(0.88, 0.96), color="#4C72B0")
for i, v in enumerate(cmp.set_index("model")["val_f1_macro"].sort_values()):
    ax.text(v + 0.001, i, f"{v:.3f}", va="center")
ax.set_xlabel("validation macro F1"); ax.set_ylabel(""); ax.set_title("Model comparison")
plt.tight_layout(); plt.savefig("reports/figures/model_comparison.png", dpi=150); plt.show()

# %% [markdown]
# ## Test results (used once)

# %%
pd.DataFrame({name: {"roles": ", ".join(r["roles"]), **r["test"]} for name, r in rep["test"].items()}).T

# %%
display(Image("reports/confusion_matrix.png")); display(Image("reports/per_class_f1.png"))

# %% [markdown]
# ## Leakage check
# The model input is only title + abstract. No CPC code or OpenAlex topic ID appears in any text.

# %%
pd.Series(rep["leakage_check"])

# %% [markdown]
# ## Embedding choice: SPECTER2 vs MiniLM
# SPECTER2 + Logistic Regression is slightly better for classification, while MiniLM is better for
# similarity retrieval (notebook 03) and is the only one that runs under fastembed on the free
# backend, so MiniLM is the production embedding.

# %%
sim = json.loads(Path("reports/similarity_eval.json").read_text())
pd.DataFrame({
    "classification val macro F1 (+ LogReg)": rep["embedding_comparison_val_f1_macro"],
    "retrieval MRR": {"minilm": sim["minilm"]["mrr"], "specter2": sim["specter2"]["mrr"]},
    "dimension": {"minilm": 384, "specter2": 768},
})

# %% [markdown]
# ## Error analysis
# Most errors sit on domain boundaries: AI methods applied to medical data (AI ↔ Healthcare),
# learning-based robot control (Robotics ↔ AI), ML for intrusion or malware detection
# (AI ↔ Cybersecurity) and robots or control systems for energy plants (Robotics ↔ Renewable
# Energy). These documents genuinely belong to two domains; the label follows the source's primary
# code or topic.

# %%
pd.Series(rep["top_confusions_best_model"], name="test errors")

# %%
pd.set_option("display.max_colwidth", 110)
pd.read_csv("reports/misclassified_examples.csv")
