# Innovation Intelligence Platform — Implementation Plan

**Nov 1, 2026 · @Beast**

## 1. Overview

Build a free-to-run web platform that collects about 30,000 patents and research papers from universities and research institutes (2020–2025), classifies them into five technology domains with a supervised model whose accuracy is measured and reported, and lets researchers search, compare and explore innovation trends.

Project topic: an intelligent system that analyzes patents, innovation projects and research outputs to identify technology domains, similarities, emerging innovation areas and relationships among institutional innovations.

Technology domains (classes): Artificial Intelligence, Healthcare, Robotics, Cybersecurity, Renewable Energy.

**Deliverables**

- A data pipeline that fetches, cleans and labels patents (Google Patents Public Datasets on BigQuery) and papers (OpenAlex).
- A supervised domain classifier, with four models compared and accuracy, precision, recall, F1 and a confusion matrix reported.
- Semantic search and similarity detection between any two innovations.
- Automatic clustering of innovations into named sub-areas.
- Emerging-technology detection from yearly growth.
- Institution comparison and an institution relationship network.
- A FastAPI backend and a React dashboard, deployed on free hosting.
- An evaluation report (metrics, plots, tables) for the project write-up.

**How to use this plan with Claude Code**

- Export this doc as Markdown and save it as PLAN.md in the project root.
- Work one phase per Claude Code session: "Read PLAN.md and complete Phase N. Stop when its acceptance checks pass and report the results."
- Review the output and commit before starting the next phase.

## 2. Agent operating rules and permissions

Claude Code has full access inside the project folder: it may create, edit and delete files there, install Python and npm packages, run scripts, notebooks, tests and servers, and make git commits without asking.

**Full access, with these boundaries**

| Area | Rule |
|---|---|
| Scope | Work only inside the project folder. Never modify or delete files outside it. |
| System software | Install any software or packages needed (system tools, CLIs, Python/npm packages) directly, without asking. Still ask before changing OS settings. |
| Secrets | Read API keys only from .env. Never print, log or commit keys. .env stays in .gitignore; keep a .env.example with empty values. |
| Git | Commit after every completed phase with a clear message (phase-3: embeddings pipeline). Never force-push or rewrite history. |
| Data | Never delete data/raw/. Re-fetching costs API quota, BigQuery scan quota and time. |
| External services | Ask before creating cloud accounts or deploying (Supabase, Render, Vercel). The user signs up and pastes keys into .env. |
| BigQuery quota | Free sandbox limit is 1 TB of queries per month. Dry-run every query and print the estimate; set maximum_bytes_billed to 200 GB per query; stop and ask if an estimate is higher. Never SELECT *. |
| Reproducibility | Fix random seeds (seed = 42). Save every model, metric and plot to disk with the config that produced it. |
| Honesty | Report real metrics only. Never tune on the test set or hand-pick results. |

**Working style**

- Read PLAN.md before each phase and follow its acceptance checks.
- Prefer simple, readable code with docstrings and type hints; a student must be able to explain it.
- Put shared settings (domains, years, paths, model names) in config.yaml, not scattered in code.
- Make long jobs resumable: save progress in batches so a crash or network drop does not lose work.
- At the end of each phase, report what was built, the commands to run it and any metrics.

**Environment**

- Laptop: Intel i7 12th gen, 16 GB RAM, NVIDIA RTX 4050 (6 GB VRAM), 100 GB free SSD.
- Python 3.11 in a virtual environment, PyTorch with CUDA, Node.js 20+ for the frontend.
- Keys needed in .env: GCP_PROJECT_ID, OPENALEX_API_KEY, GEMINI_API_KEY, DATABASE_URL.
- Google Cloud CLI installed; BigQuery credentials from `gcloud auth application-default login` (the user finishes the browser sign-in).

## 3. Architecture and tech stack

All training and analysis run once on the laptop's GPU; the online stack only serves precomputed results, which keeps hosting free.

**system architecture · offline pipeline, free online stack**

Only the user's search query is embedded live, by the lightweight fastembed model on the backend; everything else is read from the database.

| Layer | Technology | Runs on | Cost |
|---|---|---|---|
| Data collection | Python, requests, tenacity (OpenAlex); google-cloud-bigquery, db-dtypes, pyarrow (patents) | Laptop + BigQuery sandbox | Free (1 TB queries/month) |
| ML and NLP | PyTorch (CUDA), sentence-transformers, scikit-learn, transformers, BERTopic, FAISS | Laptop (RTX 4050) | Free |
| Experiments | Jupyter notebooks, matplotlib, seaborn, plotly | Laptop | Free |
| Database | PostgreSQL + pgvector | Local for development; Supabase free tier online | Free |
| Backend | FastAPI, fastembed, SQLAlchemy/psycopg | Render free web service | Free |
| LLM features | Gemini API (Google AI Studio free tier) | Called offline and on demand | Free tier |
| Frontend | React, Vite, Tailwind, Recharts/Plotly, Cytoscape.js | Vercel | Free |
| Code | Git + GitHub (Student plan) | GitHub | Free |

## 4. Repository structure

One monorepo holds the pipeline, notebooks, backend and frontend; raw data and models stay out of git.

```
innovation-platform/
├── PLAN.md                  # this plan
├── README.md                # setup + run instructions
├── config.yaml              # domains, CPC map, years, paths, model names
├── .env.example             # key names, no values
├── requirements.txt
├── data/                    # git-ignored
│   ├── raw/                 # API responses (patents/, papers/)
│   ├── processed/           # documents.parquet, embeddings.npy, splits
│   └── outputs/             # clusters, trends, institution scores
├── models/                  # git-ignored trained models
├── reports/                 # metrics JSON, plots, tables for the write-up
├── src/pipeline/
│   ├── fetch_patents.py
│   ├── fetch_papers.py
│   ├── clean.py
│   ├── embed.py
│   ├── train_classifier.py
│   ├── cluster.py
│   ├── trends.py
│   ├── institutions.py
│   ├── load_db.py
│   └── run_all.py           # whole pipeline, one command
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_classification_experiments.ipynb
│   ├── 03_similarity_evaluation.ipynb
│   ├── 04_clustering.ipynb
│   └── 05_trends_institutions.ipynb
├── backend/
│   ├── app/main.py          # FastAPI app
│   ├── app/routers/         # search, documents, trends, institutions, explain
│   ├── app/services/        # db, embedding (fastembed), gemini
│   └── tests/
└── frontend/                # React + Vite
    └── src/pages/           # Search, Document, Compare, Dashboard, Institutions, Network
```

## 5. Data sources, labels and schema

All data comes from two free sources, fetched once to local files; labels come from the sources themselves, so no manual labelling is needed.

**Sources**

| Source | Gives | Access | Limits |
|---|---|---|---|
| Google Patents Public Datasets (BigQuery table `patents-public-data.patents.publications`) | Granted patents (US and IN, configurable): title, abstract, harmonized assignee, dates, CPC codes, family ID, citations | Free BigQuery sandbox (no credit card); project ID in GCP_PROJECT_ID, credentials via `gcloud auth application-default login` | 1 TB of queries per month; tables expire after 60 days |
| OpenAlex API | Papers: title, abstract, institutions, year, topics | Free key, sent as api_key parameter | $1 of free usage per day; use per_page=100 |

Institutional innovations = patents whose assignee is a university or research institute, plus papers whose authors belong to one. Filter assignees by name keywords (university, institute, college, IIT, research). BigQuery's harmonized assignee names are abbreviated ("UNIV CALIFORNIA", "MASSACHUSETTS INST TECHNOLOGY"), so the patent filter is a word-boundary regex that includes the abbreviations (`bigquery.assignee_pattern` in config.yaml).

**Domain labels**

| Domain | Patent CPC prefixes | Paper label (OpenAlex) |
|---|---|---|
| Artificial Intelligence | G06N | Topics under Artificial Intelligence / Machine Learning |
| Healthcare | A61B, G16H | Topics under Medicine / Health informatics |
| Robotics | B25J, G05B 19 | Topics under Robotics / Control |
| Cybersecurity | G06F 21, H04L 9 | Topics under Computer security / Cryptography |
| Renewable Energy | Y02E, H02S, F03D | Topics under Renewable energy / Energy engineering |

The exact CPC prefixes and OpenAlex topic IDs live in config.yaml; Claude Code verifies them against the API in Phase 1 and records the final lists in the report.

- Patents take the label of their first (primary) CPC code (the `cpc` entry flagged `first`, else the first entry); drop patents whose primary code matches no domain.
- Patents are deduplicated by family_id: one publication per family (the US grant if there is one, else the earliest grant).
- Papers take the label of their primary topic.
- Target about 6,000 documents per domain (roughly half patents, half papers), years 2020–2025, so classes are balanced.

**Unified schema (documents.parquet)**

| Column | Type | Example |
|---|---|---|
| doc_id | string | PAT-11234567, PAP-W4389 |
| source | enum | patent, paper |
| title | string | Deep learning system for retinal disease detection |
| abstract | string | Full abstract text |
| text | string | title + ". " + abstract (model input) |
| institution | string | Normalised institution name |
| country | string | IN, US |
| year | int | 2023 |
| domain | enum | Healthcare (the label) |
| raw_codes | list | CPC codes or OpenAlex topic IDs (never a model input) |
| cited_ids | list | Cited patents or works, used for similarity evaluation |

OpenAlex returns abstracts as an inverted index; clean.py rebuilds the plain text.

**BigQuery to unified schema (patents)**

| Unified column | BigQuery source |
|---|---|
| doc_id | "PAT-" + publication_number (e.g. PAT-US-11234567-B2) |
| source | constant "patent" |
| title | title_localized, entry with language = 'en' |
| abstract | abstract_localized, entry with language = 'en' |
| institution | assignee_harmonized.name of the first assignee matching the university/institute pattern (normalised in Phase 2) |
| country | that assignee's country_code (falls back to the publication country_code) |
| year | year of grant_date |
| domain | domain whose CPC prefix matches the primary CPC code |
| raw_codes | all cpc.code values, primary first |
| cited_ids | citation.publication_number values, prefixed "PAT-" |

Extra patent columns kept for reference: publication_number, family_id, patent_office (publication country_code), grant_date.

## 6. Implementation phases

Thirteen phases run in order: data and ML first (Phases 0–7), then database, backend and frontend (8–11), then deployment and the report (12). Each phase ends with acceptance checks that must pass before the next starts.

### Phase 0 — Environment setup (week 1)

- Create the repo structure from Section 4, .gitignore, .env.example, config.yaml, README.md.
- Create a Python 3.11 virtual environment; install CUDA-enabled PyTorch, pandas, pyarrow, scikit-learn, sentence-transformers, bertopic, umap-learn, hdbscan, transformers, datasets, fastembed, requests, tenacity, python-dotenv, pyyaml, jupyter, matplotlib, seaborn, plotly.
- Write scripts/check_env.py that prints GPU name, CUDA availability and which keys are present in .env.

**Done when:** check_env.py shows the RTX 4050 with CUDA available and all API keys found.

### Phase 1 — Data collection (week 1–2)

- fetch_patents.py: query `patents-public-data.patents.publications` on BigQuery, reading only publication_number, family_id, country_code, publication_date, grant_date, title_localized, abstract_localized, cpc, assignee_harmonized and citation (never SELECT *).
  - Filters: grant_date 2020-01-01 to 2025-12-31; English title and abstract; country_code in the configured list (US, IN); primary CPC matching a domain prefix; an assignee matching the university/institute pattern.
  - Deduplicate by family_id (one publication per family).
  - Quota safety: dry-run every query and print the estimated bytes; maximum_bytes_billed = 200 GB per query; stop and ask if the estimate is higher.
  - Run ONE filtered query and save it to a small staging table in the user's sandbox dataset; draw each domain's sample from that table with cheap queries.
  - Balance to about 3,000 patents per domain; save to data/raw/patents/<domain>.parquet; reuse the staging table and existing files (resumable).
- fetch_papers.py: query OpenAlex works per domain topic, 2020–2025, with abstracts and institutions; respect rate limits (sleep + retry with backoff), save each page to data/raw/papers/ as JSON, skip pages already saved (resumable), accept --domain, --years, --max arguments.
- Log counts per domain, source, country and year.

**Done when:** about 6,000 records per domain (about 3,000 patents + 3,000 papers) exist in data/raw/, and rerunning a script fetches nothing new (no BigQuery scan on rerun).

### Phase 2 — Cleaning and labelling (week 2)

- clean.py: rebuild OpenAlex abstracts, drop records with missing or very short abstracts (< 30 words), remove duplicates by normalised title.
- Normalise institution names (lowercase, strip punctuation, map known aliases).
- Assign domain from the primary CPC code or primary topic; drop unmatched records.
- Write documents.parquet with the Section 5 schema.
- Create stratified 70/15/15 train/validation/test splits by domain and save the IDs.
- Notebook 01_data_exploration: class balance, documents per year, top institutions, abstract lengths.

**Done when:** no domain holds more than 30% of documents, and the splits are saved.

### Phase 3 — Embeddings (week 3)

- embed.py: encode text with sentence-transformers/all-MiniLM-L6-v2 on the GPU (batch 128, normalised vectors); also run allenai/specter2_base for comparison.
- Save embeddings_<model>.npy aligned to doc_id.

**Done when:** every document has a 384-dimension MiniLM vector and the run takes minutes, not hours.

### Phase 4 — Supervised classification (week 3–4)

Train and compare four models on the same splits (Section 7 lists the metrics):

- TF-IDF + Multinomial Naive Bayes (baseline).
- TF-IDF + Logistic Regression and Linear SVM.
- Sentence embeddings + Logistic Regression.
- Fine-tuned SciBERT or DistilBERT (3–4 epochs, fp16, batch 16 to fit 6 GB VRAM).

Tune on validation only. Evaluate the best model once on the test set. Save metrics to reports/classification.json, plus confusion-matrix plots and a model comparison table.

**Done when:** all four models are trained and the comparison table and test results are saved.

### Phase 5 — Similarity and semantic search (week 4)

- Build a FAISS inner-product index over the embeddings for local experiments.
- similar(doc_id, k) returns the top-k most similar documents with cosine scores.
- search(query, k, filters) embeds the query and returns top-k with optional domain, year, source and institution filters.
- Flag cross-institution pairs with similarity above a threshold chosen in the notebook.

**Done when:** Precision@10 is computed (Section 7) and example queries return sensible results.

### Phase 6 — Innovation clustering (week 5)

- cluster.py: BERTopic on the embeddings (UMAP + HDBSCAN), within each domain and across all documents.
- Save each cluster's ID, keywords, size and members.
- Compute silhouette score and NMI/ARI against domain labels.

**Done when:** clusters and their metrics are saved, and clusters such as "AI for medical imaging" are visible.

### Phase 7 — Emerging technologies and institutions (week 5–6)

- trends.py: count documents per cluster per year; compute growth rate and CAGR for 2020–2025; rank clusters by recent growth with a minimum-size threshold.
- Back-test: detect emerging clusters using 2020–2023 only, then check whether they grew in 2024–2025.
- institutions.py: per institution, documents by domain, a profile vector (mean embedding), pairwise cosine similarity, and top similar institutions.
- Build the network: nodes = institutions, edges = similarity above threshold or shared clusters; save as JSON.

**Done when:** a ranked emerging list, the back-test result, institution profiles and the network JSON are saved.

### Phase 8 — Database (week 6)

- Run local PostgreSQL with the pgvector extension (Supabase in production).
- Tables: documents (Section 5 columns + embedding vector(384) + cluster_id), clusters, trends, institutions, institution_similarity, explanations_cache.
- load_db.py loads everything; add an HNSW index on embedding.

**Done when:** a vector query returns the same top-10 as the FAISS index for test documents.

### Phase 9 — Backend API (week 7)

- FastAPI app with the Section 8 endpoints, Pydantic response models and CORS for the frontend URL.
- Query embedding at runtime with fastembed using the same MiniLM model (no PyTorch, fits 512 MB RAM).
- Serve the trained classifier for /classify (the embeddings + Logistic Regression model is small enough for free hosting).
- pytest tests for every endpoint.

**Done when:** all tests pass and /docs shows every endpoint working against the local database.

### Phase 10 — Frontend dashboard (week 8–9)

- React + Vite + Tailwind; charts with Recharts or Plotly; network with Cytoscape.js.
- Build the Section 8 pages; read the API base URL from an environment variable.
- Loading and empty states; works on a laptop and a phone screen.

**Done when:** every page loads real data from the local backend.

### Phase 11 — Gemini features (week 9)

- Offline: send each cluster's keywords and sample titles to the Gemini API to get a readable name and a 2-sentence summary; store them in clusters.
- Offline: one-paragraph summaries for the top emerging areas.
- On demand: /explain?a=&b= asks Gemini why two documents are similar, and caches the answer.
- Every Gemini call has a fallback (keywords as the name, no explanation) so the platform works without it.

**Done when:** clusters show readable names and the Explain button works with caching.

### Phase 12 — Deployment and report (week 10)

- Deploy as in Section 9 and run the smoke tests.
- Generate reports/final_report.md with all metrics, plots and tables.
- Update README.md with architecture, setup, the run order and screenshots.

**Done when:** the public URL works end to end and the report is complete.

## 7. Evaluation plan

The headline result is the domain classifier's test-set accuracy and macro F1; every other module also gets a measurable score so the whole system is evaluated.

| Module | Learning type | Metrics | Ground truth |
|---|---|---|---|
| Domain classification | Supervised | Accuracy, precision, recall, F1 (macro and weighted), confusion matrix, per-class report | Primary CPC code / primary OpenAlex topic |
| Similarity detection | Unsupervised retrieval | Precision@10, Recall@10, MRR | Same CPC subgroup, or a citation link between the two documents |
| Semantic search | Retrieval | Precision@10 on 20 hand-written test queries | Domain match of returned documents |
| Clustering | Unsupervised | Silhouette score, NMI, ARI | Domain labels (NMI/ARI only) |
| Emerging-tech detection | Time-series analysis | Hit rate of the back-test (flagged in 2020–23, grew in 2024–25) | Actual later counts |

**Classification protocol**

- Stratified 70/15/15 split with seed = 42; the same split for every model.
- Choose hyperparameters on validation; touch the test set once, for the final model only.
- Report a comparison table (model, accuracy, macro F1, training time) and the best model's confusion matrix.
- Leakage check: model input is only title + abstract; CPC codes, topic names and the domain never appear in it.
- Error analysis: list 20 misclassified examples and note patterns, such as AI-for-healthcare overlap.
- Also report SPECTER2 against MiniLM embeddings to justify the embedding choice.

**Report outputs (reports/)**

- classification.json, model_comparison.csv, confusion_matrix.png, per_class_f1.png
- similarity_eval.json, clustering_eval.json, trend_backtest.json
- final_report.md that collects all of them with a short interpretation

## 8. API endpoints and frontend pages

The backend exposes ten read-only endpoints; the frontend has six pages built on them.

**Backend endpoints (FastAPI)**

| Method | Endpoint | Returns |
|---|---|---|
| GET | /search?q=&domain=&year_from=&year_to=&source=&k= | Top-k documents for a natural-language query, with scores |
| GET | /documents/{doc_id} | One document with its domain, cluster and institution |
| GET | /documents/{doc_id}/similar?k= | Most similar documents, marking other institutions |
| GET | /compare?a=&b= | Similarity score between two documents and their shared cluster/domain |
| GET | /explain?a=&b= | Gemini explanation of why two documents are similar (cached) |
| POST | /classify | Predicted domain and probabilities for a new title + abstract |
| GET | /stats/overview | Counts by domain, source and year; top institutions |
| GET | /trends?domain= | Yearly counts and growth per cluster; emerging list |
| GET | /clusters and /clusters/{id} | Cluster names, keywords, sizes and members |
| GET | /institutions, /institutions/{name}, /institutions/network | Profiles, similar institutions, network nodes and edges |

**Frontend pages (React)**

| Page | Shows |
|---|---|
| Search | Natural-language search box, filters, result cards with domain and score |
| Document detail | Abstract, domain, cluster, similar innovations from other institutions, Explain button |
| Compare | Pick two documents; similarity score, shared topics, Gemini explanation |
| Dashboard | Most researched domains, top institutions, innovation trends over time, emerging technologies |
| Institutions | Compare up to three institutions side by side; domain mix; similar institutions |
| Network | Interactive institution graph, edges by technology similarity, filter by domain |

A "Classify new innovation" box on the Dashboard lets an examiner paste an abstract and see the model's prediction live.

## 9. Deployment, testing, risks and master prompt

The platform deploys at zero cost because all heavy computation runs once on the laptop and only precomputed results go online.

**Deployment steps**

- Supabase: create a free project, enable pgvector, run the schema, then load_db.py with the Supabase DATABASE_URL.
- Render: deploy backend/ as a free web service (start command uvicorn app.main:app --host 0.0.0.0 --port $PORT); set DATABASE_URL, GEMINI_API_KEY and ALLOWED_ORIGINS as environment variables.
- Vercel: deploy frontend/ with VITE_API_URL pointing to the Render URL.
- Smoke test: search, similar, compare, explain, dashboard and network all work on the public URL.

**Testing**

- Unit tests for cleaning (abstract rebuild, deduplication, label mapping).
- pytest for every API endpoint against a small test database.
- A 200-document sample run of run_all.py that finishes in minutes, used before full runs.

**Risks and fallbacks**

| Risk | Fallback |
|---|---|
| Render free tier sleeps after 15 minutes idle; first request takes about 30 seconds | Open the site a minute before the demo; record a backup demo video |
| Render has 512 MB RAM | Use fastembed (no PyTorch) and only the small embedding classifier online |
| API rate or quota limits slow the fetch | Resumable scripts; spread fetching over several days |
| A BigQuery query scans too much of the 1 TB/month sandbox quota | Dry-run first, 200 GB maximum_bytes_billed cap, read only the needed columns, run the big scan once into a staging table and sample from it |
| BigQuery sandbox tables expire after 60 days | Patents are saved locally to data/raw/patents/ as soon as they are sampled; the staging table is only an intermediate |
| Too few university-owned patents in a domain | Widen years to 2018–2025 or relax the assignee filter, and document the change |
| Free database storage runs short | Store 384-dimension vectors only; keep abstracts trimmed in the online copy |
| Gemini API limits or downtime | Precompute names and summaries offline; cache explanations; keyword fallback |
| Fine-tuning runs out of 6 GB VRAM | fp16, batch size 8–16, max length 256 tokens, gradient accumulation |

**Master prompt for Claude Code**

Paste this at the start of each session, changing the phase number:

```
You are the lead engineer on this project. Read PLAN.md fully before doing anything.
Complete Phase <N> exactly as written, following Section 2 (operating rules).
You have full access inside this project folder: create and edit files, install
Python/npm packages, run scripts, notebooks and tests, and commit to git.
Install any software or packages you need without asking.
Ask me before creating cloud accounts or deploying.
Read API keys only from .env; never print or commit them.
When the phase's "Done when" checks pass, commit, then report:
(1) what you built, (2) the commands to run it, (3) metrics or counts, (4) any issues.
Then stop and wait for me.
```

**Sources**

- Google Patents Public Datasets on BigQuery (patents-public-data.patents.publications)
- BigQuery sandbox limits
- OpenAlex API authentication and pricing
- Render free hosting review 2026

## 10. Changes from the original plan

Changes made during implementation, with the reason for each (Section 9: "document the change").

| Date | Change | Reason |
|---|---|---|
| 2026-11-01 | Patents come from the PatentsView **bulk files** on the USPTO Open Data Portal instead of the PatentSearch API. `.env` key `PATENTSVIEW_API_KEY` is renamed `USPTO_API_KEY` (an ODP API key). Phase 1 `fetch_patents.py` downloads and filters the bulk tables instead of paging an API. | PatentsView's PatentSearch API was shut down on 2026-03-20 and its data moved to ODP bulk datasets. |
| 2026-11-02 | **Supersedes the row above.** Patents come from Google Patents Public Datasets on BigQuery (free sandbox) instead of PatentsView / USPTO. `.env` drops the USPTO key and adds `GCP_PROJECT_ID`. Countries are US and IN (configurable). Patents are deduplicated by family_id. Section 2 now allows installing software without asking. | User decision: BigQuery needs only a Google account, while the USPTO Open Data Portal key needs ID.me identity verification. |
