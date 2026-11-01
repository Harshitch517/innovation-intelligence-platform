"""Phase 3: encode every document's text into a dense vector on the GPU.

Two models:
  minilm   sentence-transformers/all-MiniLM-L6-v2 (384-d, mean pooling). The production
           embedding: small enough for fastembed on the free backend.
  specter2 allenai/specter2_base (768-d, [CLS] pooling, title [SEP] abstract), a scientific-
           document model, run for comparison in Phase 4 / notebook 03.

Vectors are L2-normalised, so a dot product is the cosine similarity. Output rows follow the
order of data/processed/documents.parquet; the doc_id order is saved next to them.

Output: data/processed/embeddings_<model>.npy, data/processed/embedding_ids.json,
        reports/embedding_runs.json (model, dimension, seconds, device)

Usage:
    python -m src.pipeline.embed                 # both models
    python -m src.pipeline.embed --model minilm
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
import torch

from src.config import load_config, path


def embed_minilm(texts: list[str], model_name: str, batch_size: int, device: str) -> np.ndarray:
    """Sentence-transformers encoding with normalised vectors."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device=device)
    return model.encode(
        texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=True, convert_to_numpy=True
    ).astype(np.float32)


@torch.no_grad()
def embed_specter2(titles: list[str], abstracts: list[str], model_name: str, batch_size: int, device: str) -> np.ndarray:
    """SPECTER2 base: [CLS] vector of "title [SEP] abstract", fp16 on the GPU."""
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()
    if device == "cuda":
        model = model.half()
    texts = [t + tokenizer.sep_token + a for t, a in zip(titles, abstracts)]
    out = []
    for i in range(0, len(texts), batch_size):
        batch = tokenizer(texts[i : i + batch_size], padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
        cls = model(**batch).last_hidden_state[:, 0, :].float()
        out.append(torch.nn.functional.normalize(cls, dim=-1).cpu().numpy())
        if (i // batch_size) % 20 == 0:
            print(f"  specter2 {i + len(batch['input_ids']):,}/{len(texts):,}")
    return np.concatenate(out).astype(np.float32)


def main() -> None:
    """Embed all documents with the requested model(s) and log the run times."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=["minilm", "specter2", "all"], default="all")
    args = parser.parse_args()

    cfg = load_config()
    m = cfg["models"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    docs = pd.read_parquet(path("documents"))
    proc = path("processed")
    (proc / "embedding_ids.json").write_text(json.dumps(docs["doc_id"].tolist()), encoding="utf-8")

    runs_file = path("reports") / "embedding_runs.json"
    runs = json.loads(runs_file.read_text(encoding="utf-8")) if runs_file.exists() else {}
    for name in (["minilm", "specter2"] if args.model == "all" else [args.model]):
        start = time.perf_counter()
        if name == "minilm":
            vectors = embed_minilm(docs["text"].tolist(), m["embedding"], m["embedding_batch_size"], device)
        else:
            vectors = embed_specter2(docs["title"].tolist(), docs["abstract"].tolist(), m["embedding_compare"], 32, device)
        seconds = round(time.perf_counter() - start, 1)
        np.save(proc / f"embeddings_{name}.npy", vectors)
        runs[name] = {
            "model": m["embedding"] if name == "minilm" else m["embedding_compare"],
            "documents": int(vectors.shape[0]),
            "dimension": int(vectors.shape[1]),
            "seconds": seconds,
            "device": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
            "norm_check": round(float(np.linalg.norm(vectors[:100], axis=1).mean()), 4),
        }
        print(f"[{name}] {vectors.shape} in {seconds}s")
    runs_file.write_text(json.dumps(runs, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
