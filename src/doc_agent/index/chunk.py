"""Stage 4 — chunk text"""
from __future__ import annotations
import json
from pathlib import Path
from ..contracts import Chunk

CHUNK_DIR = Path("data/interim/chunks")


def _token_count(text: str) -> int:
    return len(text.split())


def split(chunks: list[Chunk], cfg: dict) -> list[Chunk]:
    """Re-chunk raw per-region OCR chunks to cfg['index'] size/overlap.

    Input: small, region-sized Chunks from ocr.transcribe() (one per detected
    region — text/table/heading blocks), already in correct multi-column
    reading order from layout.py. This function regroups them, never reorders.
    """
    index_cfg = cfg.get("index", {})
    target_tokens = index_cfg.get("chunk_tokens", 256)
    overlap_tokens = index_cfg.get("overlap", 32)

    by_doc: dict[str, list[Chunk]] = {}
    for c in chunks:
        by_doc.setdefault(c.doc_id, []).append(c)

    output: list[Chunk] = []

    for doc_id, doc_chunks in by_doc.items():
        buffer_words: list[str] = []
        buffer_page_ids: list[str] = []
        chunk_idx = 0

        def flush():
            nonlocal chunk_idx
            if not buffer_words:
                return
            text = " ".join(buffer_words)
            seen = set()
            ordered_pages = [p for p in buffer_page_ids if not (p in seen or seen.add(p))]
            output.append(Chunk(
                id=f"{doc_id}_c{chunk_idx:04d}",
                doc_id=doc_id,
                text=text,
                page_ids=ordered_pages,
            ))
            chunk_idx += 1

        for c in doc_chunks:
            words = c.text.split()
            for w in words:
                buffer_words.append(w)
                if c.page_ids:
                    buffer_page_ids.extend(c.page_ids)

                if len(buffer_words) >= target_tokens:
                    flush()
                    carry = buffer_words[-overlap_tokens:] if overlap_tokens > 0 else []
                    carry_pages = buffer_page_ids[-1:] if buffer_page_ids else []
                    buffer_words = list(carry)
                    buffer_page_ids = list(carry_pages)

        flush()

    # Checkpoint, matching layout.py/ocr.py's pattern — lets embed.py (or you,
    # re-running this notebook later) reload without recomputing.
    CHUNK_DIR.mkdir(parents=True, exist_ok=True)
    out_path = CHUNK_DIR / "chunks.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump([c.model_dump() for c in output], f, indent=2, ensure_ascii=False)
    print(f"Saved {len(output)} final chunks to {out_path}")

    return output