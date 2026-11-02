-- Innovation Intelligence Platform database schema (PostgreSQL + pgvector).
-- Used locally (Docker) and on Supabase. load_db.py runs this file, then loads the data.

CREATE EXTENSION IF NOT EXISTS vector;

DROP TABLE IF EXISTS explanations_cache, institution_edges, institution_similarity, institutions,
    trends, documents, clusters CASCADE;

CREATE TABLE clusters (
    cluster_id     text PRIMARY KEY,          -- "G-12" (global) or "HC-3" (sub-area of a domain)
    scope          text NOT NULL,             -- global | domain
    domain         text,                      -- NULL for global clusters
    keywords       text[] NOT NULL,
    size           integer NOT NULL,
    sample_titles  text[] NOT NULL,
    name           text,                      -- readable name (Gemini, Phase 11), NULL until then
    summary        text,
    emerging_summary text,                    -- one paragraph, top emerging sub-areas only
    recent_growth  real,                      -- sub-areas only (Phase 7)
    cagr           real,
    emerging_rank  integer
);

CREATE TABLE documents (
    doc_id           text PRIMARY KEY,
    source           text NOT NULL,           -- patent | paper
    title            text NOT NULL,
    abstract         text NOT NULL,
    institution      text NOT NULL,           -- normalised key
    institution_name text NOT NULL,           -- display name
    country          text,
    year             integer NOT NULL,
    domain           text NOT NULL,
    raw_codes        text[] NOT NULL,         -- CPC codes or OpenAlex topic IDs (never a model input)
    split            text NOT NULL,           -- train | val | test
    global_cluster   text REFERENCES clusters (cluster_id),
    cluster_id       text REFERENCES clusters (cluster_id),   -- sub-area within the domain
    embedding        vector(384) NOT NULL     -- MiniLM, L2-normalised
);
CREATE INDEX documents_domain_year ON documents (domain, year);
CREATE INDEX documents_institution ON documents (institution);
CREATE INDEX documents_cluster ON documents (cluster_id);

CREATE TABLE trends (
    cluster_id  text REFERENCES clusters (cluster_id),
    year        integer,
    count       integer NOT NULL,
    PRIMARY KEY (cluster_id, year)
);

CREATE TABLE institutions (
    institution   text PRIMARY KEY,
    name          text NOT NULL,
    country       text,
    documents     integer NOT NULL,
    patents       integer NOT NULL,
    papers        integer NOT NULL,
    top_domain    text NOT NULL,
    by_domain     jsonb NOT NULL,
    by_year       jsonb NOT NULL,
    top_clusters  text[] NOT NULL,
    similar_institutions text[] NOT NULL   -- top 5 by profile similarity
);

CREATE TABLE institution_similarity (
    institution          text REFERENCES institutions (institution),
    similar_institution  text REFERENCES institutions (institution),
    similarity           real NOT NULL,
    PRIMARY KEY (institution, similar_institution)
);

CREATE TABLE institution_edges (
    source           text REFERENCES institutions (institution),
    target           text REFERENCES institutions (institution),
    similarity       real NOT NULL,
    shared_clusters  integer NOT NULL,
    PRIMARY KEY (source, target)
);

CREATE TABLE explanations_cache (
    doc_a       text,
    doc_b       text,
    explanation text NOT NULL,
    model       text NOT NULL,
    created_at  timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (doc_a, doc_b)
);
