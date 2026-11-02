# Innovation Intelligence Platform

INT395 project. The platform analyses about 30,000 patents and research papers from universities and research institutes (2020–2025). It classifies them into five technology domains: Artificial Intelligence, Healthcare, Robotics, Cybersecurity and Renewable Energy. Researchers can use it to search, compare and explore innovation trends.

The full plan, phases and evaluation protocol are in [PLAN.md](PLAN.md).

## Architecture

All training and analysis run once, offline, on a laptop GPU. The online stack only serves the precomputed results, so hosting stays free.

| Layer | Technology | Runs on |
|---|---|---|
| Data collection | Python, requests, tenacity | Laptop |
| ML and NLP | PyTorch (CUDA), sentence-transformers, scikit-learn, transformers, BERTopic, FAISS | Laptop (RTX 4050) |
| Database | PostgreSQL + pgvector | Docker locally; Supabase online |
| Backend | FastAPI, fastembed | Render (free) |
| LLM features | Gemini API | Offline + on demand |
| Frontend | React, Vite, Tailwind, Recharts, Cytoscape.js | Vercel (free) |

## Data sources

| Source | Gives | Key |
|---|---|---|
| [Google Patents Public Datasets](https://console.cloud.google.com/marketplace/product/google_patents_public_datasets/google-patents-public-data) on BigQuery | Granted US and IN patents: title, abstract, assignee, dates, CPC codes, family, citations | `GCP_PROJECT_ID` (free BigQuery sandbox, no card) + `gcloud auth application-default login` |
| [OpenAlex](https://openalex.org/) | Papers: title, abstract, institutions, year, topics | `OPENALEX_API_KEY` (free account) |
| [Google AI Studio](https://aistudio.google.com/apikey) | Gemini cluster names and explanations | `GEMINI_API_KEY` |

## Setup

Requirements: Python 3.11, an NVIDIA GPU with CUDA, Node.js 20+, Docker (for the local database) and the Google Cloud CLI (for BigQuery).

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt

copy .env.example .env      # then paste your keys and GCP project ID into .env
gcloud auth application-default login   # BigQuery credentials (browser sign-in)
python scripts/check_env.py # shows GPU, CUDA and which keys are present
```

## Run the pipeline

```powershell
python -m src.pipeline.fetch_papers            # OpenAlex papers -> data/raw/papers/
python -m src.pipeline.fetch_patents --dry-run # BigQuery size estimate only
python -m src.pipeline.fetch_patents           # BigQuery patents -> data/raw/patents/
python -m src.pipeline.clean                   # -> data/processed/documents.parquet + splits
python -m src.pipeline.embed                   # MiniLM + SPECTER2 vectors (GPU)
python -m src.pipeline.train_classifier        # 4 model families, reports/classification.json
python -m src.pipeline.similarity              # FAISS index, retrieval + search evaluation
python -m src.pipeline.cluster                 # BERTopic sub-areas
python -m src.pipeline.trends                  # emerging list + back-test
python -m src.pipeline.institutions            # profiles, similarity, network
docker run -d --name innovation-db -e POSTGRES_PASSWORD=postgres -e POSTGRES_DB=innovation -p 5432:5432 -v innovation_pgdata:/var/lib/postgresql/data pgvector/pgvector:pg17
python -m src.pipeline.load_db                 # schema + data + HNSW index, FAISS check
python -m pytest                               # unit tests
python scripts/build_notebook.py notebooks/src/01_data_exploration.py notebooks/01_data_exploration.ipynb
```

Notebooks are written as plain Python in `notebooks/src/` and built and executed into `notebooks/*.ipynb`
by `scripts/build_notebook.py`.

## Project status

| Phase | Status |
|---|---|
| 0 — Environment setup | done |
| 1 — Data collection | done (patents: BigQuery, papers: OpenAlex) |
| 2 — Cleaning and labelling | done (30,000 docs, 6,000 per domain) |
| 3 — Embeddings | done (MiniLM 384-d in 134 s, SPECTER2 768-d in 239 s) |
| 4 — Supervised classification | pending |
| 5 — Similarity and semantic search | done (search P@10 0.97, MiniLM MRR 0.21) |
| 6 — Innovation clustering | done (94 global clusters, 161 sub-areas, NMI 0.48) |
| 7 — Emerging technologies and institutions | done (605 institutions, back-test reported) |
| 8 — Database | done (pgvector HNSW top-10 = FAISS for 100/100 test docs) |
| 9 — Backend API | pending |
| 10 — Frontend dashboard | pending |
| 11 — Gemini features | pending |
| 12 — Deployment and report | pending |

## Repository layout

```
config.yaml        shared settings (domains, CPC map, years, paths, models)
src/pipeline/      fetch, clean, embed, train, cluster, trends, institutions, load_db
notebooks/         exploration and evaluation notebooks
backend/           FastAPI app and tests
frontend/          React + Vite dashboard
reports/           metrics, plots and tables for the write-up
data/, models/     generated locally, git-ignored
```
