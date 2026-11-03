"""Read an uploaded Word/PDF/text document and pick out its title and abstract.

Everything here runs locally (pypdf, python-docx, regular expressions): no LLM call, so
analysing an upload costs no Gemini credits. Only the title + abstract are used afterwards.
"""
from __future__ import annotations

import io
import re

MAX_BYTES = 5 * 1024 * 1024
MAX_PDF_PAGES = 30
ABSTRACT_WORDS = 300

HEADING_RE = re.compile(r"^\s*(abstract|summary|abstract of the disclosure|summary of the invention)\s*[:.\-–—]?\s*(.*)$", re.I)
END_RE = re.compile(r"^\s*(keywords?|index terms|1\.?\s+introduction|introduction|background|field of the invention|"
                    r"technical field|claims?|what is claimed|description|brief description of the drawings|references)\b", re.I)


class ExtractionError(ValueError):
    """The file cannot be read or has no usable text."""


def _lines_from_pdf(data: bytes) -> tuple[list[str], str | None, int]:
    from pypdf import PdfReader

    try:
        reader = PdfReader(io.BytesIO(data))
    except Exception as e:  # noqa: BLE001 - any parse failure is a bad upload
        raise ExtractionError("This PDF could not be opened.") from e
    if reader.is_encrypted:
        raise ExtractionError("This PDF is password-protected.")
    pages = reader.pages[:MAX_PDF_PAGES]
    text = "\n".join((p.extract_text() or "") for p in pages)
    meta_title = (reader.metadata.title if reader.metadata else None) or None
    return text.splitlines(), meta_title, len(reader.pages)


def _lines_from_docx(data: bytes) -> tuple[list[str], str | None, int]:
    import docx

    try:
        d = docx.Document(io.BytesIO(data))
    except Exception as e:  # noqa: BLE001
        raise ExtractionError("This Word file could not be opened (only .docx is supported, not old .doc).") from e
    title = next((p.text for p in d.paragraphs if p.text.strip() and p.style.name.lower().startswith(("title", "heading"))), None)
    return [p.text for p in d.paragraphs], title or d.core_properties.title or None, 0


def _clean(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def _good_title(t: str | None) -> bool:
    return bool(t) and 3 <= len(t.split()) <= 30 and not re.match(r"^(microsoft word|untitled|document\d*)", t.strip(), re.I)


def extract(filename: str, data: bytes) -> dict:
    """Return {title, abstract, word_count, pages, file_type, method} for an uploaded file."""
    if len(data) > MAX_BYTES:
        raise ExtractionError("File is larger than 5 MB.")
    name = filename.lower()
    if name.endswith(".pdf"):
        lines, meta_title, pages, ftype = (*_lines_from_pdf(data), "pdf")
    elif name.endswith(".docx"):
        lines, meta_title, pages, ftype = (*_lines_from_docx(data), "docx")
    elif name.endswith(".txt"):
        lines, meta_title, pages, ftype = data.decode("utf-8", errors="ignore").splitlines(), None, 0, "txt"
    else:
        raise ExtractionError("Please upload a .pdf, .docx or .txt file.")

    lines = [_clean(l) for l in lines]
    lines = [l for l in lines if l]
    words = sum(len(l.split()) for l in lines)
    if words < 20:
        raise ExtractionError("No readable text found (a scanned PDF needs OCR first).")

    title = _clean(meta_title) if _good_title(meta_title) else next((l for l in lines[:15] if _good_title(l)), lines[0][:200])

    # Abstract: the text under an "Abstract"/"Summary" heading, up to the next section heading.
    abstract, method = "", "first_words"
    for i, line in enumerate(lines):
        m = HEADING_RE.match(line)
        if not m:
            continue
        parts = [m.group(2)] if m.group(2) else []
        for nxt in lines[i + 1:]:
            if END_RE.match(nxt) or sum(len(p.split()) for p in parts) >= ABSTRACT_WORDS:
                break
            parts.append(nxt)
        candidate = _clean(" ".join(parts))
        if len(candidate.split()) >= 25:
            abstract, method = candidate, "abstract_section"
            break
    if not abstract:  # fall back to the opening text after the title
        body = [l for l in lines if l != title]
        abstract = _clean(" ".join(body))

    abstract = " ".join(abstract.split()[:ABSTRACT_WORDS])
    return {"title": title[:300], "abstract": abstract, "word_count": words, "pages": pages,
            "file_type": ftype, "method": method}
