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
```

## Project status

| Phase | Status |
|---|---|
| 0 — Environment setup | done |
| 1 — Data collection | done (patents: BigQuery, papers: OpenAlex) |
| 2 — Cleaning and labelling | pending |
| 3 — Embeddings | pending |
| 4 — Supervised classification | pending |
| 5 — Similarity and semantic search | pending |
| 6 — Innovation clustering | pending |
| 7 — Emerging technologies and institutions | pending |
| 8 — Database | pending |
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
