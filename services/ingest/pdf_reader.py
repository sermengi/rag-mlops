from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional
import datetime as dt
import hashlib

from services.ingest.normalize.cleaner import normalize_text
from pypdf import PdfReader

@dataclass(frozen=True)
class PageText:
    page_no: int
    raw_text: str
    text: str # normalized; used for chunking

@dataclass(frozen=True)
class RawDoc:
    doc_id: str               # stable identifier you choose upstream
    source_type: str          # "pdf"
    source_value: str         # absolute path (file:) or storage key
    mime: str                 # "application/pdf"
    fetched_at: str           # ISO timestamp
    content_hash: str         # SHA256 of raw bytes (idempotency)
    meta: Dict[str, str]      # filename/title/author/etc. (best-effort)
    pages: List[PageText]     # per-page text


def _sha256(b: bytes) -> str:
    h = hashlib.sha256()
    h.update(b)
    return h.hexdigest()

def load_pdf(path: str | Path, *, doc_id: Optional[str] = None) -> RawDoc:
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"PDF not found: {p}")

    raw_bytes = p.read_bytes()
    content_hash = _sha256(raw_bytes)

    if doc_id is None:
        doc_id = f"{p.stem}:{content_hash[:8]}"

    reader = PdfReader(p)
    pages: List[PageText] = []
    for i, page in enumerate(reader.pages, start=1):
        try:
            raw_text = page.extract_text() or ""
            text = normalize_text(raw_text)
            pages.append(PageText(page_no=i, raw_text=raw_text ,text=text))
        except Exception as e:
            raise e

    normalized_pages = [normalize_text(p.text) for p in pages]

    meta = {
        "filename": p.name,
        "title": (reader.metadata.title or "").strip() if reader.metadata else "",
        "author": (reader.metadata.author or "").strip() if reader.metadata else "",
        "content_hash_raw": content_hash,
        "normalized_hash": _sha256("\n\n".join(normalized_pages).encode("utf-8")),
        "normalization_version": "v1",
    }

    return RawDoc(
        doc_id=doc_id,
        source_type="pdf",
        source_value=str(p),
        mime="application/pdf",
        fetched_at=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds") + "Z",
        content_hash=content_hash,
        meta=meta,
        pages=pages,
    )

if __name__ == "__main__":
    import sys
    import json
    if len(sys.argv) < 2:
        print("Usage: python pdf_reader.py /path/to/file.pdf")
        raise SystemExit(2)

    rd = load_pdf(sys.argv[1])
    preview = {
        "doc_id": rd.doc_id,
        "hash": rd.content_hash[:12],
        "pages": len(rd.pages),
        "title": rd.meta.get("title", ""),
        "author": rd.meta.get("author", ""),
        "sample_page_1": (rd.pages[0].text[:400] + "...") if rd.pages else "",
    }

    print(json.dumps(preview, ensure_ascii=False, indent=2))
