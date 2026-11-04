"""Endpoint tests against the local database (load it first with `python -m src.pipeline.load_db`)."""
from app.services import gemini
from app.services.db import execute

DOMAINS = {"Artificial Intelligence", "Healthcare", "Robotics", "Cybersecurity", "Renewable Energy"}


def test_health(client):
    r = client.get("/")
    assert r.status_code == 200 and r.json()["documents"] >= 30000


def test_search_returns_ranked_relevant_results(client):
    r = client.get("/search", params={"q": "perovskite solar cell efficiency", "k": 10})
    body = r.json()
    assert r.status_code == 200 and body["count"] == 10
    scores = [x["score"] for x in body["results"]]
    assert scores == sorted(scores, reverse=True)
    assert sum(x["domain"] == "Renewable Energy" for x in body["results"]) >= 8


def test_search_filters(client):
    r = client.get("/search", params={"q": "robot gripper", "domain": "Robotics", "source": "patent",
                                      "year_from": 2023, "year_to": 2024, "k": 15})
    res = r.json()["results"]
    assert len(res) == 15
    assert all(x["domain"] == "Robotics" and x["source"] == "patent" and 2023 <= x["year"] <= 2024 for x in res)


def test_search_validation(client):
    assert client.get("/search", params={"q": "x"}).status_code == 422
    assert client.get("/search", params={"q": "valid query", "source": "book"}).status_code == 422


def test_document_detail_and_404(client, sample):
    r = client.get(f"/documents/{sample['patent']}")
    d = r.json()
    assert r.status_code == 200 and d["domain"] in DOMAINS and d["cluster"]["cluster_id"] == sample["cluster"]
    assert len(d["abstract"]) > 50
    assert client.get("/documents/PAT-NOPE").status_code == 404


def test_similar_marks_other_institutions(client, sample):
    r = client.get(f"/documents/{sample['patent']}/similar", params={"k": 5})
    res = r.json()["results"]
    assert len(res) == 5 and all(x["doc_id"] != sample["patent"] for x in res)
    assert all(x["other_institution"] == (x["institution"] != sample_institution(client, sample["patent"])) for x in res)
    only = client.get(f"/documents/{sample['patent']}/similar", params={"k": 5, "other_institutions_only": True}).json()
    assert all(x["other_institution"] for x in only["results"])


def sample_institution(client, doc_id):
    return client.get(f"/documents/{doc_id}").json()["institution"]


def test_compare(client, sample):
    r = client.get("/compare", params={"a": sample["patent"], "b": sample["paper"]})
    body = r.json()
    assert r.status_code == 200 and -1 <= body["similarity"] <= 1
    assert isinstance(body["shared_keywords"], list)
    same = client.get("/compare", params={"a": sample["patent"], "b": sample["patent"]}).json()
    assert same["similarity"] > 0.999 and same["same_cluster"]
    assert client.get("/compare", params={"a": sample["patent"], "b": "PAP-NOPE"}).status_code == 404


def test_explain_fallback_and_cache(client, sample, monkeypatch):
    a, b = sorted((sample["patent"], sample["paper"]))
    execute("DELETE FROM explanations_cache WHERE doc_a = %s AND doc_b = %s", (a, b))
    monkeypatch.setattr(gemini, "generate", lambda *args, **kw: None)
    r = client.get("/explain", params={"a": a, "b": b}).json()
    assert r["explanation"] is None and r["note"] and not r["cached"]

    monkeypatch.setattr(gemini, "generate", lambda *args, **kw: "They share a method.")
    first = client.get("/explain", params={"a": b, "b": a}).json()  # order does not matter
    second = client.get("/explain", params={"a": a, "b": b}).json()
    assert first["explanation"] == "They share a method." and not first["cached"]
    assert second["cached"] and second["explanation"] == "They share a method."
    execute("DELETE FROM explanations_cache WHERE doc_a = %s AND doc_b = %s", (a, b))


def test_classify(client):
    r = client.post("/classify", json={
        "title": "Intrusion detection for IoT networks",
        "abstract": "We propose an anomaly-based intrusion detection system that monitors network traffic "
                    "of IoT devices and detects attacks such as botnets and denial of service in real time."})
    body = r.json()
    assert r.status_code == 200 and body["domain"] == "Cybersecurity"
    assert abs(sum(body["probabilities"].values()) - 1) < 1e-3 and set(body["probabilities"]) == DOMAINS
    assert client.post("/classify", json={"title": "x", "abstract": "short"}).status_code == 422


def test_overview(client):
    body = client.get("/stats/overview").json()
    assert body["documents"] >= 30000 and body["patents"] + body["papers"] == body["documents"]
    assert {x["key"] for x in body["by_domain"]} == DOMAINS
    assert len(body["top_institutions"]) == 15


def test_trends(client):
    body = client.get("/trends", params={"domain": "Healthcare"}).json()
    assert set(body["domain_counts"]) == {"2020", "2021", "2022", "2023", "2024", "2025"}
    assert body["clusters"] and all(c["domain"] == "Healthcare" for c in body["clusters"])
    ranks = [c["emerging_rank"] for c in body["emerging"]]
    assert ranks == sorted(ranks) and len(body["emerging"]) <= 10
    assert all(c["counts"] for c in body["emerging"])


def test_clusters(client, sample):
    allc = client.get("/clusters").json()
    assert {c["scope"] for c in allc} == {"global", "domain"}
    detail = client.get(f"/clusters/{sample['cluster']}").json()
    assert detail["members"] and detail["counts"] and detail["top_institutions"]
    g = client.get("/clusters", params={"scope": "global"}).json()[0]
    assert client.get(f"/clusters/{g['cluster_id']}").json()["members"]
    assert client.get("/clusters/XX-999").status_code == 404


def test_institutions(client, sample):
    lst = client.get("/institutions", params={"limit": 5}).json()
    assert len(lst) == 5 and lst[0]["documents"] >= lst[-1]["documents"]
    found = client.get("/institutions", params={"q": "tsinghua"}).json()
    assert any("Tsinghua" in x["name"] for x in found)
    d = client.get(f"/institutions/{sample['institution']}").json()
    assert d["similar"] and d["recent_documents"] and sum(d["by_domain"].values()) == d["documents"]
    assert client.get("/institutions/not a real place").status_code == 404


def test_network(client):
    full = client.get("/institutions/network").json()
    assert len(full["nodes"]) > 500 and len(full["edges"]) > 1000
    ids = {n["id"] for n in full["nodes"]}
    assert all(e["source"] in ids and e["target"] in ids for e in full["edges"])
    robo = client.get("/institutions/network", params={"domain": "Robotics"}).json()
    assert 0 < len(robo["nodes"]) < len(full["nodes"])


def test_lpu_profile_and_search(client):
    found = client.get("/institutions", params={"q": "lovely"}).json()
    assert found and found[0]["name"] == "Lovely Professional University" and found[0]["documents"] >= 300
    d = client.get(f"/institutions/{found[0]['institution']}").json()
    assert d["country"] == "IN" and d["similar"] and d["recent_documents"]
    res = client.get("/search", params={"q": "intrusion detection", "institution": found[0]["institution"], "k": 5}).json()
    assert len(res["results"]) == 5 and all(r["institution_name"] == "Lovely Professional University" for r in res["results"])


def test_full_text_links(client, sample, monkeypatch):
    from app.services import fulltext

    r = client.get(f"/documents/{sample['patent']}/fulltext").json()
    assert r["kind"] == "patent" and r["url"].startswith("https://patents.google.com/patent/US") and not r["embeddable"]
    assert client.get(f"/documents/{sample['patent']}/pdf").status_code == 404
    assert client.get("/documents/PAP-NOPE/fulltext").status_code == 404

    links = {"url": "https://x.org/a.pdf", "pdf_url": "https://x.org/a.pdf", "is_open_access": True, "embeddable": True,
             "authors": ["A. Author"], "author_count": 1, "venue": "A Journal", "cited_by_count": 3}
    monkeypatch.setattr(fulltext, "paper_links", lambda w: links)
    body = client.get(f"/documents/{sample['paper']}/fulltext").json()
    assert body["embeddable"] and body["authors"] == ["A. Author"] and body["venue"] == "A Journal"
    monkeypatch.setattr(fulltext, "stream_pdf", lambda url: (iter([b"%PDF-1.4 test"]), 13))
    r = client.get(f"/documents/{sample['paper']}/pdf")
    assert r.status_code == 200 and r.headers["content-type"] == "application/pdf" and r.content.startswith(b"%PDF")
    assert r.headers["content-length"] == "13"
    monkeypatch.setattr(fulltext, "stream_pdf", lambda url: None)
    assert client.get(f"/documents/{sample['paper']}/pdf").status_code == 404
