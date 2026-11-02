"""Phase 4: train and compare domain classifiers (PLAN.md Section 7, classification protocol).

Models, all trained on the same train split and tuned on the validation split only:
  1. TF-IDF + Multinomial Naive Bayes (baseline)
  2. TF-IDF + Logistic Regression
  3. TF-IDF + Linear SVM
  4. Sentence embeddings + Logistic Regression (MiniLM, and SPECTER2 for comparison)
  5. Fine-tuned SciBERT (fp16, batch 16, max 256 tokens; best epoch chosen on validation)

The model input is only `text` (title + abstract). The test split is touched once, after every
choice is frozen: for the best model by validation macro F1, and for the embeddings + Logistic
Regression model that the API serves (/classify).

Outputs (reports/): classification.json, model_comparison.csv, confusion_matrix.png,
per_class_f1.png, misclassified_examples.csv. Models go to models/.

Usage:
    python -m src.pipeline.train_classifier             # all models
    python -m src.pipeline.train_classifier --skip-bert # classic models only
"""
from __future__ import annotations

import argparse
import json
import random
import re
import time
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
import torch  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.naive_bayes import MultinomialNB  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.svm import LinearSVC  # noqa: E402

from src.config import load_config, path  # noqa: E402

CPC_RE = re.compile(r"\b[A-H]\d{2}[A-Z]\s?\d{1,4}/\d{2,6}\b")


def set_seed(seed: int) -> None:
    """Fix every random seed used here."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_data() -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Documents joined with their split, plus the embedding matrices in the same row order."""
    docs = pd.read_parquet(path("documents"))
    splits = pd.read_parquet(path("processed") / "splits.parquet")
    docs = docs.merge(splits, on="doc_id", how="left", validate="one_to_one")
    ids = json.loads((path("processed") / "embedding_ids.json").read_text(encoding="utf-8"))
    assert ids == docs["doc_id"].tolist(), "embeddings are not aligned with documents.parquet"
    emb = {name: np.load(path("processed") / f"embeddings_{name}.npy") for name in ("minilm", "specter2")}
    return docs, emb


def leakage_check(docs: pd.DataFrame, cfg: dict) -> dict:
    """Confirm labels and label sources do not appear in the model input."""
    text = docs["text"]
    topic_ids = {t for spec in cfg["domains"].values() for t in spec["openalex_topics"]}
    return {
        "model_input": "text = title + '. ' + abstract (no other columns)",
        "docs_with_cpc_code_in_text": int(text.str.contains(CPC_RE).sum()),
        "docs_with_openalex_topic_id_in_text": int(text.str.contains("|".join(sorted(topic_ids))).sum()),
        "note": "domain, raw_codes and topic names are never passed to any model",
    }


def scores(y_true, y_pred, labels) -> dict:
    """Accuracy and macro / weighted precision, recall and F1."""
    out = {"accuracy": accuracy_score(y_true, y_pred)}
    for avg in ("macro", "weighted"):
        p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, average=avg, zero_division=0)
        out |= {f"precision_{avg}": p, f"recall_{avg}": r, f"f1_{avg}": f}
    return {k: round(float(v), 4) for k, v in out.items()}


# ---------------------------------------------------------------- classic models

def tfidf(**kw) -> TfidfVectorizer:
    """Shared TF-IDF settings: unigrams + bigrams, sublinear tf, English stopwords."""
    return TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.9, sublinear_tf=True, stop_words="english", **kw)


def classic_candidates(seed: int) -> dict[str, list[tuple[dict, object]]]:
    """Model family -> list of (hyperparameters, unfitted estimator) to try on validation."""
    return {
        "TF-IDF + Naive Bayes": [
            ({"alpha": a}, Pipeline([("tfidf", tfidf(max_features=200_000)), ("clf", MultinomialNB(alpha=a))]))
            for a in (0.01, 0.05, 0.1, 0.5)
        ],
        "TF-IDF + Logistic Regression": [
            ({"C": c}, Pipeline([("tfidf", tfidf(max_features=200_000)), ("clf", LogisticRegression(C=c, max_iter=2000, random_state=seed))]))
            for c in (1, 4, 16)
        ],
        "TF-IDF + Linear SVM": [
            ({"C": c}, Pipeline([("tfidf", tfidf(max_features=200_000)), ("clf", LinearSVC(C=c, random_state=seed))]))
            for c in (0.1, 0.5, 1.0)
        ],
    }


def embedding_candidates(seed: int) -> list[tuple[dict, object]]:
    """Logistic Regression settings to try on the sentence embeddings."""
    return [({"C": c}, LogisticRegression(C=c, max_iter=3000, random_state=seed)) for c in (1, 4, 16, 64)]


def tune(candidates, x_train, y_train, x_val, y_val) -> tuple[dict, object, float, float, list[dict]]:
    """Fit each candidate on train, keep the one with the best validation macro F1."""
    best, trials = None, []
    for params, model in candidates:
        start = time.perf_counter()
        model.fit(x_train, y_train)
        seconds = time.perf_counter() - start
        f1 = f1_score(y_val, model.predict(x_val), average="macro")
        trials.append({"params": params, "val_f1_macro": round(float(f1), 4), "train_seconds": round(seconds, 1)})
        if best is None or f1 > best[2]:
            best = (params, model, f1, seconds)
    return (*best, trials)


# ---------------------------------------------------------------- SciBERT

class TextDataset(torch.utils.data.Dataset):
    """Tokenised texts with integer labels."""

    def __init__(self, encodings: dict, labels: list[int]) -> None:
        self.encodings, self.labels = encodings, labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i: int) -> dict:
        item = {k: torch.tensor(v[i]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[i])
        return item


@torch.no_grad()
def bert_predict(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    """Predicted class ids and probabilities for every batch in `loader`."""
    model.eval()
    probs = []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items() if k != "labels"}
        with torch.autocast("cuda", dtype=torch.float16, enabled=device == "cuda"):
            logits = model(**batch).logits
        probs.append(torch.softmax(logits.float(), dim=-1).cpu().numpy())
    p = np.concatenate(probs)
    return p.argmax(1), p


def train_bert(docs: pd.DataFrame, labels: list[str], cfg: dict, out_dir: Path, epochs: int = 3) -> dict:
    """Fine-tune SciBERT on train; keep the epoch with the best validation macro F1."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

    device = "cuda" if torch.cuda.is_available() else "cpu"
    name = cfg["models"]["transformer_classifier"]
    tok = AutoTokenizer.from_pretrained(name)
    lid = {l: i for i, l in enumerate(labels)}

    def loader(split: str, shuffle: bool) -> torch.utils.data.DataLoader:
        part = docs[docs["split"] == split]
        enc = tok(part["text"].tolist(), truncation=True, max_length=256, padding="max_length")
        ds = TextDataset(dict(enc), [lid[d] for d in part["domain"]])
        return torch.utils.data.DataLoader(ds, batch_size=16, shuffle=shuffle, generator=torch.Generator().manual_seed(cfg["seed"]))

    train_dl, val_dl = loader("train", True), loader("val", False)
    y_val = docs.loc[docs["split"] == "val", "domain"].map(lid).to_numpy()
    model = AutoModelForSequenceClassification.from_pretrained(name, num_labels=len(labels)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=0.01)
    steps = epochs * len(train_dl)
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")

    history, best_f1, start = [], -1.0, time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for step, batch in enumerate(train_dl, 1):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast("cuda", dtype=torch.float16, enabled=device == "cuda"):
                loss = model(**batch).loss
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            total += loss.item()
            if step % 300 == 0:
                print(f"  scibert epoch {epoch} step {step}/{len(train_dl)} loss {total / step:.4f}")
        pred, _ = bert_predict(model, val_dl, device)
        f1 = f1_score(y_val, pred, average="macro")
        history.append({"epoch": epoch, "train_loss": round(total / len(train_dl), 4), "val_f1_macro": round(float(f1), 4)})
        print(f"  scibert epoch {epoch}: val macro F1 {f1:.4f}")
        if f1 > best_f1:
            best_f1 = f1
            model.save_pretrained(out_dir)
            tok.save_pretrained(out_dir)
    return {"val_f1_macro": float(best_f1), "train_seconds": time.perf_counter() - start, "history": history,
            "params": {"lr": 3e-5, "epochs": epochs, "batch_size": 16, "max_length": 256, "fp16": device == "cuda"}}


def bert_test_predictions(docs: pd.DataFrame, labels: list[str], model_dir: Path, split: str = "test") -> tuple[np.ndarray, np.ndarray]:
    """Load the saved best SciBERT and predict one split (test by default)."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
    part = docs[docs["split"] == split]
    enc = tok(part["text"].tolist(), truncation=True, max_length=256, padding="max_length")
    ds = TextDataset(dict(enc), [0] * len(part))
    pred, probs = bert_predict(model, torch.utils.data.DataLoader(ds, batch_size=64), device)
    return np.array(labels)[pred], probs


# ---------------------------------------------------------------- reports

def plot_confusion(y_true, y_pred, labels, title: str, file: Path) -> list[list[int]]:
    """Save a row-normalised confusion matrix plot; return the raw counts."""
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    norm = cm / cm.sum(axis=1, keepdims=True)
    plt.figure(figsize=(7, 5.5))
    sns.heatmap(norm, annot=cm, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, cbar_kws={"label": "row share"})
    plt.xlabel("predicted"); plt.ylabel("true"); plt.title(title)
    plt.xticks(rotation=30, ha="right"); plt.tight_layout(); plt.savefig(file, dpi=150); plt.close()
    return cm.tolist()


def plot_per_class_f1(report: dict, labels: list[str], title: str, file: Path) -> None:
    """Bar chart of per-class F1 from a classification_report dict."""
    f1 = pd.Series({l: report[l]["f1-score"] for l in labels}).sort_values()
    ax = f1.plot.barh(figsize=(7, 3.5), color="#4C72B0", xlim=(min(0.8, f1.min() - 0.02), 1.0))
    for i, v in enumerate(f1):
        ax.text(v + 0.002, i, f"{v:.3f}", va="center")
    ax.set_xlabel("F1 (test)"); ax.set_title(title)
    plt.tight_layout(); plt.savefig(file, dpi=150); plt.close()


def main() -> None:
    """Train, tune, compare, test the frozen choices once and write every report file."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--skip-bert", action="store_true", help="skip SciBERT fine-tuning")
    args = parser.parse_args()

    cfg = load_config()
    set_seed(cfg["seed"])
    docs, emb = load_data()
    labels = sorted(cfg["domains"])
    reports, models_dir = path("reports"), path("models")
    mask = {s: (docs["split"] == s).to_numpy() for s in ("train", "val", "test")}
    y = {s: docs.loc[mask[s], "domain"].to_numpy() for s in mask}
    text = {s: docs.loc[mask[s], "text"].tolist() for s in mask}

    results, fitted = {}, {}
    for family, cands in classic_candidates(cfg["seed"]).items():
        print(f"Training {family} ...")
        params, model, f1, secs, trials = tune(cands, text["train"], y["train"], text["val"], y["val"])
        results[family] = {"params": params, "val": scores(y["val"], model.predict(text["val"]), labels), "train_seconds": round(secs, 1), "trials": trials}
        fitted[family] = ("text", model)
        print(f"  best {params}: val macro F1 {f1:.4f}")

    for name, label in (("minilm", "MiniLM embeddings + Logistic Regression"), ("specter2", "SPECTER2 embeddings + Logistic Regression")):
        print(f"Training {label} ...")
        x = {s: emb[name][mask[s]] for s in mask}
        params, model, f1, secs, trials = tune(embedding_candidates(cfg["seed"]), x["train"], y["train"], x["val"], y["val"])
        results[label] = {"params": params, "val": scores(y["val"], model.predict(x["val"]), labels), "train_seconds": round(secs, 1), "trials": trials}
        fitted[label] = (name, model)
        print(f"  best {params}: val macro F1 {f1:.4f}")

    bert_name = "Fine-tuned SciBERT"
    if not args.skip_bert:
        print(f"Training {bert_name} ...")
        bert_dir = models_dir / "scibert_classifier"
        bert = train_bert(docs, labels, cfg, bert_dir)
        val_pred = bert_test_predictions(docs, labels, bert_dir, split="val")[0]
        results[bert_name] = {"params": bert["params"], "val": scores(y["val"], val_pred, labels),
                              "train_seconds": round(bert["train_seconds"], 1), "history": bert["history"]}
        fitted[bert_name] = ("bert", bert_dir)

    # Every choice is frozen here. The test split is used once, below.
    best_name = max(results, key=lambda k: results[k]["val"]["f1_macro"])
    served_name = "MiniLM embeddings + Logistic Regression"

    def test_predict(name: str):
        kind, model = fitted[name]
        if kind == "bert":
            return bert_test_predictions(docs, labels, model)[0]
        return model.predict(text["test"] if kind == "text" else emb[kind][mask["test"]])

    test_results = {}
    for role, name in (("best_model", best_name), ("served_model", served_name)):
        if name in test_results:
            continue
        pred = test_predict(name)
        test_results[name] = {"roles": [], "test": scores(y["test"], pred, labels),
                              "per_class": classification_report(y["test"], pred, labels=labels, output_dict=True, zero_division=0),
                              "pred": pred}
    for role, name in (("best_model", best_name), ("served_model", served_name)):
        test_results[name]["roles"].append(role)

    # reports
    best = test_results[best_name]
    cm = plot_confusion(y["test"], best["pred"], labels, f"{best_name} — test confusion matrix", reports / "confusion_matrix.png")
    plot_per_class_f1(best["per_class"], labels, f"{best_name} — per-class F1 (test)", reports / "per_class_f1.png")
    if served_name != best_name:
        plot_confusion(y["test"], test_results[served_name]["pred"], labels, f"{served_name} — test", reports / "confusion_matrix_served.png")

    test_docs = docs[mask["test"]].reset_index(drop=True)
    wrong = test_docs[best["pred"] != y["test"]].assign(predicted=best["pred"][best["pred"] != y["test"]])
    wrong.sample(n=min(20, len(wrong)), random_state=cfg["seed"])[["doc_id", "source", "domain", "predicted", "title"]].to_csv(
        reports / "misclassified_examples.csv", index=False)
    confusions = wrong.groupby(["domain", "predicted"]).size().sort_values(ascending=False)

    comparison = pd.DataFrame([
        {"model": k, "val_accuracy": v["val"].get("accuracy"), "val_f1_macro": v["val"]["f1_macro"],
         "train_seconds": v["train_seconds"], "params": json.dumps(v["params"])}
        for k, v in results.items()
    ]).sort_values("val_f1_macro", ascending=False)
    comparison.to_csv(reports / "model_comparison.csv", index=False)

    for name, (kind, model) in fitted.items():
        if kind != "bert":
            slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
            joblib.dump(model, models_dir / f"{slug}.joblib")
    joblib.dump({"model": fitted[served_name][1], "labels": labels, "embedding": cfg["models"]["embedding"]},
                models_dir / "served_classifier.joblib")

    out = {
        "config": {"seed": cfg["seed"], "splits": cfg["splits"], "labels": labels,
                   "n": {s: int(m.sum()) for s, m in mask.items()}},
        "leakage_check": leakage_check(docs, cfg),
        "validation": results,
        "best_model_by_val_f1_macro": best_name,
        "served_model": served_name,
        "test": {k: {kk: vv for kk, vv in v.items() if kk != "pred"} for k, v in test_results.items()},
        "best_model_confusion_matrix": {"labels": labels, "counts": cm},
        "top_confusions_best_model": {f"{a} -> {b}": int(n) for (a, b), n in confusions.head(8).items()},
        "embedding_comparison_val_f1_macro": {
            "minilm": results["MiniLM embeddings + Logistic Regression"]["val"]["f1_macro"],
            "specter2": results["SPECTER2 embeddings + Logistic Regression"]["val"]["f1_macro"],
        },
    }
    (reports / "classification.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(comparison[["model", "val_accuracy", "val_f1_macro", "train_seconds"]].to_string(index=False))
    for name, r in test_results.items():
        print(f"TEST {name} ({', '.join(r['roles'])}): accuracy {r['test']['accuracy']:.4f}, macro F1 {r['test']['f1_macro']:.4f}")


if __name__ == "__main__":
    main()
