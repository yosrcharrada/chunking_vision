"""Extract plain text from an uploaded document.

Supported:
  * PDF                      -> PyMuPDF (fitz)
  * DOCX                     -> python-docx
  * HTML/HTM                 -> BeautifulSoup (text only)
  * everything else (code,   -> decoded as UTF-8 / latin-1 text
    .txt, .md, .py, .json …)

Returns (text, meta) where meta records the detected kind and any notes.
"""

from __future__ import annotations

import io
import os
import re
from typing import Tuple

CODE_EXTS = {
    ".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".c", ".h", ".cpp", ".cc",
    ".hpp", ".cs", ".go", ".rs", ".rb", ".php", ".swift", ".kt", ".scala",
    ".sh", ".bash", ".sql", ".r", ".m", ".lua", ".pl",
}
TEXT_EXTS = {".txt", ".md", ".rst", ".csv", ".tsv", ".json", ".yaml", ".yml",
             ".xml", ".ini", ".cfg", ".toml", ".log"}


def _ext(name: str) -> str:
    return os.path.splitext(name or "")[1].lower()


def extract_text(filename: str, data: bytes) -> Tuple[str, dict]:
    ext = _ext(filename)
    meta = {"filename": filename, "ext": ext, "bytes": len(data), "kind": "text"}

    if ext == ".pdf":
        meta["kind"] = "pdf"
        return _from_pdf(data), meta
    if ext in (".docx",):
        meta["kind"] = "docx"
        return _from_docx(data), meta
    if ext in (".html", ".htm"):
        meta["kind"] = "html"
        return _from_html(data), meta
    if ext in (".doc",):
        # Legacy binary .doc isn't supported by python-docx; try best-effort text.
        meta["kind"] = "doc"
        meta["note"] = "Legacy .doc — extracted as raw text; consider converting to .docx/PDF."
        return _from_bytes_text(data), meta

    meta["kind"] = "code" if ext in CODE_EXTS else "text"
    return _from_bytes_text(data), meta


def _from_pdf(data: bytes) -> str:
    import fitz  # PyMuPDF

    pages = []
    with fitz.open(stream=data, filetype="pdf") as doc:
        for page in doc:
            txt = _page_text_columns(page)
            if len(txt.strip()) < 20:           # likely a scanned/image page
                ocr = _ocr_page(page)
                if ocr.strip():
                    txt = ocr
            pages.append(txt)
    pages = _strip_repeated_lines(pages)
    return _clean("\n".join(pages))


def _page_text_columns(page) -> str:
    """Read a PDF page in correct READING ORDER, handling 2-column layouts.

    PyMuPDF's plain text can interleave columns (left line, right line, …) which
    scrambles meaning.  We pull text *blocks*, detect a two-column split, and
    emit the full left column before the right one; otherwise we sort blocks
    top-to-bottom, left-to-right."""
    blocks = [b for b in page.get_text("blocks") if len(b) >= 5 and b[4].strip()]
    if not blocks:
        return ""
    mid = page.rect.width / 2.0
    left = [b for b in blocks if b[2] <= mid * 1.05]   # block right edge left of centre
    right = [b for b in blocks if b[0] >= mid * 0.95]  # block left edge right of centre
    if len(left) >= 2 and len(right) >= 2 and (len(left) + len(right)) >= 0.7 * len(blocks):
        left.sort(key=lambda b: (b[1], b[0]))
        right.sort(key=lambda b: (b[1], b[0]))
        ordered = left + right
    else:
        ordered = sorted(blocks, key=lambda b: (round(b[1] / 3.0), b[0]))
    return "\n".join(b[4].strip() for b in ordered)


def _ocr_page(page) -> str:
    """OCR a scanned page if pytesseract + Pillow are available (optional)."""
    try:
        import io

        import pytesseract
        from PIL import Image

        pix = page.get_pixmap(dpi=200)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        return pytesseract.image_to_string(img)
    except Exception:
        return ""  # OCR not installed or failed — degrade gracefully


_PAGENUM_RE = re.compile(r"^\s*(page\s*)?[\divxlcIVXLC]{1,6}\s*$", re.IGNORECASE)
_PAGEOF_RE = re.compile(r"^\s*page\s+\d+\s*(/|of|sur)\s*\d+\s*$", re.IGNORECASE)


def _strip_repeated_lines(pages: list[str]) -> list[str]:
    """Remove running headers/footers and page numbers.

    A short line that appears (verbatim or as a digit-normalised template) on a
    large fraction of pages is boilerplate (e.g. "Journal Officiel ... 21 août
    2015", "N° 67", "Page 1887").  Drop those plus standalone page-number lines.
    """
    if len(pages) < 3:
        return [_drop_pagenums(p) for p in pages]

    from collections import Counter

    def norm(line: str) -> str:
        # collapse digits so "Page 1887" and "Page 1888" count as the same header
        return re.sub(r"\d+", "#", line.strip().lower())

    counts: Counter = Counter()
    for p in pages:
        seen = set()
        for ln in p.splitlines():
            s = ln.strip()
            if not s or len(s) > 90:
                continue
            key = norm(s)
            if key not in seen:
                counts[key] += 1
                seen.add(key)

    n = len(pages)
    boiler = {k for k, c in counts.items() if c >= max(3, int(0.5 * n))}

    out = []
    for p in pages:
        kept = []
        for ln in p.splitlines():
            s = ln.strip()
            if not s:
                kept.append(ln)
                continue
            if norm(s) in boiler:
                continue
            if _PAGENUM_RE.match(s) or _PAGEOF_RE.match(s):
                continue
            kept.append(ln)
        out.append("\n".join(kept))
    return out


def _drop_pagenums(p: str) -> str:
    return "\n".join(
        ln for ln in p.splitlines()
        if not (_PAGENUM_RE.match(ln.strip()) or _PAGEOF_RE.match(ln.strip()))
    )


def _from_docx(data: bytes) -> str:
    import docx

    d = docx.Document(io.BytesIO(data))
    blocks = [p.text for p in d.paragraphs]
    # include table cell text
    for tbl in d.tables:
        for row in tbl.rows:
            blocks.append(" | ".join(c.text for c in row.cells))
    return _clean("\n".join(b for b in blocks if b is not None))


def _from_html(data: bytes) -> str:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(_decode(data), "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    return _clean(soup.get_text("\n"))


def _from_bytes_text(data: bytes) -> str:
    return _clean(_decode(data))


def _decode(data: bytes) -> str:
    for enc in ("utf-8", "utf-16", "latin-1"):
        try:
            return data.decode(enc)
        except (UnicodeDecodeError, LookupError):
            continue
    return data.decode("utf-8", errors="replace")


def _clean(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Collapse runs of blank lines; strip trailing spaces per line.
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
