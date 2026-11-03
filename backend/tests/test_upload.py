"""Upload analysis: local extraction from .docx/.pdf/.txt and the upload endpoints (no real LLM calls)."""
import io

import docx

from app.services import gemini
from app.services.db import execute
from app.services.extract import ExtractionError, extract

TITLE = "Perovskite solar cell with improved moisture stability"
ABSTRACT = ("A perovskite solar cell is disclosed in which a passivation layer of fluorinated ammonium salts "
            "covers the halide perovskite film. The layer suppresses moisture ingress and ion migration, so the "
            "device keeps more than ninety percent of its initial power conversion efficiency after one thousand "
            "hours of damp heat testing while maintaining high open circuit voltage.")


def make_docx() -> bytes:
    d = docx.Document()
    d.add_heading(TITLE, level=0)
    d.add_paragraph("Jane Doe, University of Somewhere")
    d.add_paragraph("Abstract")
    d.add_paragraph(ABSTRACT)
    d.add_paragraph("1. Introduction")
    d.add_paragraph("Solar energy matters. " * 30)
    buf = io.BytesIO()
    d.save(buf)
    return buf.getvalue()


def make_pdf() -> bytes:
    """A minimal one-page PDF with real text (built by hand, no extra dependency)."""
    lines = [TITLE, "Abstract", *[ABSTRACT[i:i + 90] for i in range(0, len(ABSTRACT), 90)], "Introduction", "Text."]
    ops = "BT /F1 10 Tf 40 800 Td 14 TL " + " ".join(f"({l.replace('(', '').replace(')', '')}) Tj T*" for l in lines) + " ET"
    objs = ["<< /Type /Catalog /Pages 2 0 R >>", "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>",
            f"<< /Length {len(ops)} >>\nstream\n{ops}\nendstream", "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"]
    out, offsets = b"%PDF-1.4\n", []
    for i, o in enumerate(objs, 1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n{o}\nendobj\n".encode()
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
    out += "".join(f"{o:010d} 00000 n \n" for o in offsets).encode()
    out += f"trailer << /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF".encode()
    return out


def test_extract_docx_finds_title_and_abstract_section():
    info = extract("paper.docx", make_docx())
    assert info["title"] == TITLE and info["method"] == "abstract_section"
    assert info["abstract"].startswith("A perovskite solar cell") and "Introduction" not in info["abstract"]


def test_extract_pdf():
    info = extract("paper.pdf", make_pdf())
    assert info["file_type"] == "pdf" and info["method"] == "abstract_section"
    assert "perovskite" in info["abstract"].lower() and "passivation" in info["abstract"].lower()


def test_extract_rejects_bad_files():
    for name, data in (("x.exe", b"MZ"), ("x.txt", b"too short"), ("x.docx", b"not a zip")):
        try:
            extract(name, data)
        except ExtractionError:
            continue
        raise AssertionError(f"{name} should be rejected")


def test_analyze_endpoint(client):
    r = client.post("/analyze", files={"file": ("paper.docx", make_docx(), "application/octet-stream")}, params={"k": 5})
    body = r.json()
    assert r.status_code == 200 and body["title"] == TITLE
    assert body["prediction"]["domain"] == "Renewable Energy"
    assert len(body["matches"]) == 5 and body["matches"][0]["snippet"]
    assert sum(m["domain"] == "Renewable Energy" for m in body["matches"]) >= 4
    bad = client.post("/analyze", files={"file": ("x.exe", b"MZ", "application/octet-stream")})
    assert bad.status_code == 422 and "pdf" in bad.json()["detail"].lower()


def test_compare_and_explain_upload(client, monkeypatch):
    match = client.post("/analyze", files={"file": ("p.txt", f"{TITLE}\nAbstract\n{ABSTRACT}".encode(), "text/plain")},
                        params={"k": 1}).json()["matches"][0]
    body = {"title": TITLE, "abstract": ABSTRACT, "doc_id": match["doc_id"]}
    cmp = client.post("/compare/upload", json=body).json()
    assert abs(cmp["similarity"] - match["score"]) < 0.01 and cmp["predicted_domain"] == "Renewable Energy"

    calls = []
    monkeypatch.setattr(gemini, "generate", lambda prompt, **kw: calls.append((prompt, kw)) or "Both passivate perovskite films.")
    from app.routers.upload import _upload_id
    execute("DELETE FROM explanations_cache WHERE doc_a = %s", (_upload_id(TITLE, ABSTRACT),))
    first = client.post("/explain/upload", json=body).json()
    second = client.post("/explain/upload", json=body).json()
    assert first["explanation"] == second["explanation"] and second["cached"] and len(calls) == 1
    assert calls[0][1]["max_tokens"] <= 250 and len(calls[0][0]) < 2600  # short prompt, capped answer
    assert first["model"] is None
    execute("DELETE FROM explanations_cache WHERE doc_a = %s", (_upload_id(TITLE, ABSTRACT),))


def test_compare_and_explain_two_uploads(client, monkeypatch):
    other = {"title": "Wind turbine blade pitch control", "abstract": "A controller adjusts the blade pitch of a wind "
             "turbine to maximise captured power and reduce loads under turbulent wind speeds and gusts."}
    body = {"a": {"title": TITLE, "abstract": ABSTRACT}, "b": other}
    same = client.post("/compare/texts", json={"a": body["a"], "b": body["a"]}).json()
    diff = client.post("/compare/texts", json=body).json()
    assert same["similarity"] > 0.999 and diff["similarity"] < same["similarity"]
    assert diff["predicted_domain_a"] == "Renewable Energy"
    calls = []
    monkeypatch.setattr(gemini, "generate", lambda prompt, **kw: calls.append(1) or "Both concern renewable power.")
    swapped = {"a": other, "b": body["a"]}
    first = client.post("/explain/texts", json=body).json()
    second = client.post("/explain/texts", json=swapped).json()  # order does not matter for the cache
    assert first["explanation"] and second["cached"] and len(calls) == 1
    from app.routers.upload import _upload_id
    execute("DELETE FROM explanations_cache WHERE doc_a = ANY(%s)", ([_upload_id(TITLE, ABSTRACT), _upload_id(other["title"], other["abstract"])],))


def test_local_insights_without_llm(client, monkeypatch):
    monkeypatch.setattr(gemini, "generate", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no LLM call expected")))
    other = {"title": "Stable perovskite photovoltaic module with passivation", "abstract": "Perovskite solar cells degrade "
             "under humidity. We passivate the perovskite film with an organic ammonium layer, which improves moisture "
             "stability. Encapsulated modules keep their power conversion efficiency over long outdoor tests."}
    ins = client.post("/compare/texts", json={"a": {"title": TITLE, "abstract": ABSTRACT}, "b": other}).json()["insights"]
    assert any("perovskite" in t for t in ins["shared_terms"]) and ins["passages"]
    assert ins["summary"].startswith("Both documents deal with") and "%" in ins["summary"]
    assert 0 < ins["term_overlap"] < 1
