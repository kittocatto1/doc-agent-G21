"""Data — data schema/quality validation at ingest"""
from __future__ import annotations
import os
import fitz  # PyMuPDF
from ..contracts import Page

MIN_PAGES = 300
MIN_WORDS = 60000
RAW_DIR = "data/raw"


def _estimate_words_for_doc(doc_id: str, raw_dir: str = RAW_DIR) -> int:
    """Rough word count using the PDF's existing embedded text layer (from IA's own
    ABBYY OCR), purely for a corpus-size sanity check. This is NOT how our own OCR
    quality is measured — that's done separately against grading_kit/labels.jsonl."""
    pdf_path = os.path.join(raw_dir, f"{doc_id}.pdf")
    if not os.path.exists(pdf_path):
        return 0
    doc = fitz.open(pdf_path)
    total = sum(len(page.get_text().split()) for page in doc)
    doc.close()
    return total


def validate(pages: list[Page]) -> None:
    """Assert min pages/words, format, no leakage across splits."""

    # --- 1. Page count ---
    n_pages = len(pages)
    assert n_pages >= MIN_PAGES, (
        f"Corpus has {n_pages} pages, below the required floor of {MIN_PAGES}."
    )

    # --- 2. Format checks ---
    for p in pages:
        assert os.path.exists(p.image_path), f"Missing image file: {p.image_path}"
        assert p.image_path.lower().endswith((".png", ".jpg", ".jpeg")), (
            f"Unexpected image format: {p.image_path}"
        )
        assert p.doc_id, f"Page {p.id} has no doc_id"

    # --- 3. Word count (via IA's embedded text layer, see note above) ---
    doc_ids = sorted({p.doc_id for p in pages})
    total_words = sum(_estimate_words_for_doc(d) for d in doc_ids)
    assert total_words >= MIN_WORDS, (
        f"Estimated {total_words} words across {len(doc_ids)} docs, "
        f"below the required floor of {MIN_WORDS}."
    )

    # --- 4. Split / leakage check ---
    # BYTE Retriever uses pretrained models only (no fine-tuning), per the A1
    # large-corpus exemption — there is no train/test split to check leakage on.
    # grading_kit/heldout_pages/ instead holds a small set never used to TUNE the
    # pipeline (prompts/thresholds), which is a different guarantee than a
    # train/test split and is not something validate() can enforce automatically.
    pass

    print(f"Validation passed: {n_pages} pages, ~{total_words} words across {len(doc_ids)} docs.")