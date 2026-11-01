"""Unit tests for Phase 2 cleaning: abstract rebuild, deduplication, label mapping, names."""
import pandas as pd

from src.config import load_config
from src.pipeline.clean import (
    balance,
    clean_text,
    filter_and_dedupe,
    make_splits,
    normalise_institution,
    paper_record,
    rebuild_abstract,
    topic_to_domain,
)
from src.pipeline.fetch_patents import cpc_condition, domain_case

CFG = load_config()


def test_rebuild_abstract_orders_words_by_position():
    index = {"deep": [0], "learning": [1, 4], "is": [2], "about": [3]}
    assert rebuild_abstract(index) == "deep learning is about learning"
    assert rebuild_abstract(None) == ""


def test_clean_text_strips_tags_entities_and_label():
    assert clean_text("<jats:p>Abstract: Solar &amp; wind  power</jats:p>") == "Solar & wind power"


def test_filter_and_dedupe_drops_short_and_duplicate_titles():
    long = " ".join(["word"] * 40)
    df = pd.DataFrame(
        {
            "doc_id": ["PAT-1", "PAP-W1", "PAP-W2", "PAP-W3"],
            "source": ["patent", "paper", "paper", "paper"],
            "title": ["Robot Arm!", "robot arm", "Another title", "Short one"],
            "abstract": [long, long, long, "too short"],
        }
    )
    log: dict = {}
    out = filter_and_dedupe(df, 30, log)
    assert sorted(out["doc_id"]) == ["PAP-W2", "PAT-1"]  # patent kept over duplicate paper
    assert log == {"dropped_short_or_missing_abstract": 1, "dropped_duplicate_title": 1}


def test_paper_label_comes_from_primary_topic_only():
    topic_domain = topic_to_domain(CFG)
    ai_topic = CFG["domains"]["Artificial Intelligence"]["openalex_topics"][0]
    work = {
        "id": "https://openalex.org/W1",
        "title": "A paper",
        "publication_year": 2022,
        "abstract_inverted_index": {"text": [0]},
        "primary_topic": {"id": f"https://openalex.org/{ai_topic}"},
        "topics": [{"id": f"https://openalex.org/{ai_topic}"}, {"id": "https://openalex.org/T99999"}],
        "authorships": [{"institutions": [{"display_name": "Acme Corp", "type": "company"},
                                          {"display_name": "IIT Delhi", "type": "education", "country_code": "IN"}]}],
        "referenced_works": ["https://openalex.org/W2"],
    }
    rec = paper_record(work, topic_domain)
    assert rec["domain"] == "Artificial Intelligence"
    assert rec["institution_raw"] == "IIT Delhi" and rec["country"] == "IN"
    assert rec["raw_codes"][0] == ai_topic and rec["cited_ids"] == ["PAP-W2"]
    work["primary_topic"] = {"id": "https://openalex.org/T99999"}
    assert paper_record(work, topic_domain) is None


def test_patent_cpc_prefix_matching():
    assert cpc_condition("G06N") == "STARTS_WITH(primary_cpc, 'G06N')"
    assert cpc_condition("H04L9") == "STARTS_WITH(primary_cpc, 'H04L9/')"  # not H04L93
    case = domain_case(CFG["domains"])
    assert "THEN 'Robotics'" in case and "'G05B19/'" in case


def test_institution_names_match_across_sources():
    n = lambda s: normalise_institution(s, CFG)  # noqa: E731
    assert n("UNIV TSINGHUA") == n("Tsinghua University")
    assert n("MASSACHUSETTS INST TECHNOLOGY") == n("Massachusetts Institute of Technology") == n("MIT")
    assert n("SEOUL NAT UNIV R&DB FOUND") == n("Seoul National University")
    assert n("UNIV LELAND STANFORD JUNIOR") == n("Stanford University")
    assert n("") == ""


def test_balance_and_splits_are_stratified():
    rows = []
    for d in ["A", "B"]:
        rows += [{"doc_id": f"PAT-{d}{i}", "source": "patent", "domain": d} for i in range(30)]
        rows += [{"doc_id": f"PAP-{d}{i}", "source": "paper", "domain": d} for i in range(80)]
    df = balance(pd.DataFrame(rows), max_patents=20, target=60, seed=42)
    assert df.groupby(["domain", "source"]).size().to_dict() == {
        ("A", "paper"): 40, ("A", "patent"): 20, ("B", "paper"): 40, ("B", "patent"): 20}
    splits = make_splits(df, {"train": 0.7, "val": 0.15, "test": 0.15}, seed=42)
    counts = splits.merge(df, on="doc_id").groupby(["split", "domain"]).size()
    assert counts["train"]["A"] == counts["train"]["B"] == 42
    assert splits["doc_id"].is_unique and len(splits) == len(df)
