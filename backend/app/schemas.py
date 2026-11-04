"""Pydantic response models for every endpoint."""
from __future__ import annotations

from pydantic import BaseModel, Field


class DocumentSummary(BaseModel):
    doc_id: str
    source: str
    title: str
    institution: str
    institution_name: str
    country: str | None
    year: int
    domain: str
    cluster_id: str | None = None


class ScoredDocument(DocumentSummary):
    score: float
    snippet: str | None = None
    other_institution: bool | None = None


class SearchResponse(BaseModel):
    query: str
    count: int
    results: list[ScoredDocument]


class ClusterBrief(BaseModel):
    cluster_id: str
    name: str | None
    keywords: list[str]


class DocumentDetail(DocumentSummary):
    abstract: str
    raw_codes: list[str]
    cluster: ClusterBrief | None
    global_cluster: ClusterBrief | None


class FullTextLinks(BaseModel):
    kind: str                  # paper | patent
    url: str | None            # best page to read the document in full
    is_open_access: bool
    embeddable: bool           # the PDF can be shown inline via /documents/{id}/pdf
    oa_status: str | None = None
    doi: str | None = None
    venue: str | None = None
    publication_date: str | None = None
    work_type: str | None = None
    authors: list[str] = []
    author_count: int = 0
    cited_by_count: int | None = None
    references_count: int | None = None


class SimilarResponse(BaseModel):
    doc_id: str
    results: list[ScoredDocument]


class Passage(BaseModel):
    a: str
    b: str
    score: float


class Insights(BaseModel):
    """Brief local comparison (no LLM)."""
    summary: str
    shared_terms: list[str]
    only_a: list[str]
    only_b: list[str]
    term_overlap: float
    passages: list[Passage]


class CompareResponse(BaseModel):
    a: DocumentSummary
    b: DocumentSummary
    similarity: float
    same_domain: bool
    same_cluster: bool
    same_global_cluster: bool
    shared_keywords: list[str]
    different_institutions: bool
    insights: Insights


class ExplainResponse(BaseModel):
    a: str
    b: str
    similarity: float
    explanation: str | None
    cached: bool
    model: str | None
    note: str | None = None


class ClassifyRequest(BaseModel):
    title: str = Field(..., min_length=3, max_length=500)
    abstract: str = Field(..., min_length=20, max_length=10_000)


class ClassifyResponse(BaseModel):
    domain: str
    probabilities: dict[str, float]
    model: str


class Count(BaseModel):
    key: str
    count: int


class InstitutionCount(BaseModel):
    institution: str
    name: str
    count: int


class OverviewResponse(BaseModel):
    documents: int
    patents: int
    papers: int
    institutions: int
    by_domain: list[Count]
    by_source: list[Count]
    by_year: list[Count]
    by_domain_year: list[dict]
    top_institutions: list[InstitutionCount]


class ClusterSummary(BaseModel):
    cluster_id: str
    scope: str
    domain: str | None
    name: str | None
    summary: str | None
    emerging_summary: str | None = None
    keywords: list[str]
    size: int
    recent_growth: float | None
    cagr: float | None
    emerging_rank: int | None


class ClusterTrend(ClusterSummary):
    counts: dict[int, int]


class TrendsResponse(BaseModel):
    domain: str | None
    domain_counts: dict[int, int]
    clusters: list[ClusterTrend]
    emerging: list[ClusterTrend]


class ClusterDetail(ClusterSummary):
    sample_titles: list[str]
    counts: dict[int, int]
    members: list[DocumentSummary]
    top_institutions: list[InstitutionCount]


class InstitutionSummary(BaseModel):
    institution: str
    name: str
    country: str | None
    documents: int
    patents: int
    papers: int
    top_domain: str


class SimilarInstitution(BaseModel):
    institution: str
    name: str
    similarity: float


class InstitutionDetail(InstitutionSummary):
    by_domain: dict[str, int]
    by_year: dict[str, int]
    top_clusters: list[ClusterBrief]
    similar: list[SimilarInstitution]
    recent_documents: list[DocumentSummary]


class NetworkNode(BaseModel):
    id: str
    name: str
    country: str | None
    documents: int
    top_domain: str


class NetworkEdge(BaseModel):
    source: str
    target: str
    similarity: float
    shared_clusters: int


class NetworkResponse(BaseModel):
    domain: str | None
    nodes: list[NetworkNode]
    edges: list[NetworkEdge]


class UploadText(BaseModel):
    """Title and abstract taken from an uploaded document (sent back by the client)."""
    title: str = Field(..., min_length=1, max_length=500)
    abstract: str = Field(..., min_length=20, max_length=4000)


class AnalyzeResponse(BaseModel):
    filename: str
    file_type: str
    pages: int
    word_count: int
    method: str
    title: str
    abstract: str
    prediction: ClassifyResponse
    matches: list[ScoredDocument]


class UploadCompareRequest(UploadText):
    doc_id: str
    upload_side: str = Field("a", pattern="^[ab]$")  # which side of the page the upload is on


class UploadCompareResponse(BaseModel):
    b: DocumentSummary
    similarity: float
    predicted_domain: str
    same_domain: bool
    insights: Insights


class UploadExplainRequest(UploadText):
    doc_id: str


class TextsRequest(BaseModel):
    """Two uploaded documents (title + abstract each)."""
    a: UploadText
    b: UploadText


class TextsCompareResponse(BaseModel):
    similarity: float
    predicted_domain_a: str
    predicted_domain_b: str
    same_domain: bool
    insights: Insights
