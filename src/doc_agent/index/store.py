"""Stage 4 — vector store"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import faiss
from ..contracts import Chunk

INDEX_DIR = Path("data/index")


def build(chunks: list[Chunk], vectors: np.ndarray, cfg: dict) -> None:
    """Persist a vector index (cfg['index']['type'])."""
    index_cfg = cfg.get("index", {})
    dim = vectors.shape[1]

    # HNSW ("Hierarchical Navigable Small World") — an approximate index.
    # M = how many links each vector keeps to its neighbors in the graph.
    # Bigger M = more accurate search, slower to build, bigger file on disk.
    m = index_cfg.get("hnsw_m", 32)

    index = faiss.IndexHNSWFlat(dim, m)

    # efConstruction: how hard the graph tries to find good neighbors while
    # BUILDING the index. Higher = better quality index, slower one-time build.
    index.hnsw.efConstruction = index_cfg.get("ef_construction", 200)

    # efSearch: how hard it searches at QUERY time. This is your main lever
    # for the scalability NFR — higher efSearch = more accurate but slower
    # per query; lower = faster but risks missing the true best match.
    # Tune this against your P95 <= 1.5s target directly.
    index.hnsw.efSearch = index_cfg.get("ef_search", 64)

    vectors = np.ascontiguousarray(vectors.astype("float32"))
    index.add(vectors)

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_DIR / "index.faiss"))

    # FAISS only stores vectors + an internal integer id (0, 1, 2...) — it
    # knows nothing about your Chunk objects. We save this side-table so
    # retriever.py can map "FAISS returned internal id 47" back to a real
    # Chunk (its text, doc_id, page_ids) for citations.
    metadata = [c.model_dump() for c in chunks]
    with open(INDEX_DIR / "chunks_meta.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    # These are exactly the numbers your A2 Section 4 report needs.
    size_mb = (INDEX_DIR / "index.faiss").stat().st_size / (1024 ** 2)
    print(f"✅ Built HNSW index: {len(chunks)} chunks, dim={dim}, "
          f"M={m}, size={size_mb:.2f} MB → {INDEX_DIR / 'index.faiss'}")


def load(cfg: dict):
    """Load the persisted index + its chunk metadata."""
    index_path = INDEX_DIR / "index.faiss"
    meta_path = INDEX_DIR / "chunks_meta.json"

    if not index_path.exists():
        raise FileNotFoundError(f"No index found at {index_path} — run build() first.")

    index = faiss.read_index(str(index_path))

    # efSearch doesn't get saved with the index — reset it here in case
    # something changed in cfg since build() ran.
    index_cfg = cfg.get("index", {})
    index.hnsw.efSearch = index_cfg.get("ef_search", 64)

    with open(meta_path, "r", encoding="utf-8") as f:
        chunk_dicts = json.load(f)

    chunks = [Chunk(**d) for d in chunk_dicts]

    print(f"Loaded index: {index.ntotal} vectors, dim={index.d}")
    return index, chunks