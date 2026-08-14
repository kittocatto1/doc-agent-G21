"""Stage 4 — vector store"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import faiss
from ..contracts import Chunk

INDEX_DIR = Path("data/index")


def build(chunks: list[Chunk], vectors: np.ndarray, cfg: dict) -> None:
    """Incrementally add new vectors to the existing HNSW index.

    New issues are appended to the existing FAISS index instead of rebuilding
    the entire vector database.

    Safety checks:
    1. Number of chunks must equal number of vectors.
    2. A document/issue (doc_id) must not already exist in the index.
       This prevents accidental duplicate indexing if the same issue is
       processed twice.
    """

    index_cfg = cfg.get("index", {})

    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    index_path = INDEX_DIR / "index.faiss"
    meta_path = INDEX_DIR / "chunks_meta.json"

    # ---------------------------------------------------------
    # 1. Basic chunk/vector consistency check
    # ---------------------------------------------------------
    #
    # FAISS assigns vector IDs based on their position:
    #
    #   FAISS ID 0 -> chunks_meta[0]
    #   FAISS ID 1 -> chunks_meta[1]
    #   ...
    #
    # Therefore these two lists MUST have the same length.
    # ---------------------------------------------------------
    if len(chunks) != len(vectors):
        raise ValueError(
            f"Chunk/vector mismatch: "
            f"{len(chunks)} chunks but {len(vectors)} vectors."
        )

    if len(chunks) == 0:
        print("No chunks supplied. Nothing to add.")
        return

    vectors = np.ascontiguousarray(vectors.astype("float32"))
    dim = vectors.shape[1]

    # ---------------------------------------------------------
    # 2. Load existing metadata and detect duplicate issues
    # ---------------------------------------------------------
    #
    # chunks_meta.json contains the metadata for every vector
    # currently stored in FAISS.
    #
    # We use doc_id to determine whether an entire issue has
    # already been indexed.
    # ---------------------------------------------------------
    if meta_path.exists():
        with meta_path.open("r", encoding="utf-8") as f:
            existing_metadata = json.load(f)
    else:
        existing_metadata = []

    existing_doc_ids = {
        item["doc_id"]
        for item in existing_metadata
    }

    new_doc_ids = {
        c.doc_id
        for c in chunks
    }

    duplicate_doc_ids = new_doc_ids & existing_doc_ids

    if duplicate_doc_ids:
        raise ValueError(
            "\n❌ Duplicate issue detected!\n"
            f"The following doc_id(s) are already indexed:\n"
            + "\n".join(
                f"  - {doc_id}"
                for doc_id in sorted(duplicate_doc_ids)
            )
            + "\n\nRefusing to modify the FAISS index."
            "\nThis prevents accidental duplicate indexing."
        )

    # ---------------------------------------------------------
    # 3. Load existing FAISS index OR create a new one
    # ---------------------------------------------------------
    m = index_cfg.get("hnsw_m", 32)
    ef_construction = index_cfg.get("ef_construction", 200)
    ef_search = index_cfg.get("ef_search", 64)

    if index_path.exists():

        print(f"Loading existing FAISS index from {index_path}")

        index = faiss.read_index(str(index_path))

        # The embedding model must produce the same dimensionality
        # as the vectors already stored in the index.
        if index.d != dim:
            raise ValueError(
                f"Embedding dimension mismatch: "
                f"existing index has dim={index.d}, "
                f"but new vectors have dim={dim}."
            )

        print(
            f"Existing index contains {index.ntotal} vectors."
        )
        print(
            f"Adding {len(vectors)} vectors from "
            f"{len(new_doc_ids)} new issue(s)."
        )

    else:

        print(
            "No existing FAISS index found. "
            "Creating a new HNSW index."
        )

        index = faiss.IndexHNSWFlat(
            dim,
            m,
            faiss.METRIC_INNER_PRODUCT,
        )

        # Controls how thoroughly HNSW searches for good neighbors
        # while constructing the graph.
        index.hnsw.efConstruction = ef_construction

    # efSearch controls search effort at query time.
    index.hnsw.efSearch = ef_search

    # ---------------------------------------------------------
    # 4. Add ONLY the new vectors
    # ---------------------------------------------------------
    #
    # Existing vectors remain untouched.
    #
    # Example:
    #
    # First run:
    #   FAISS = Issue A
    #
    # Second run:
    #   FAISS = Issue A + Issue B
    #
    # Third run:
    #   FAISS = Issue A + Issue B + Issue C
    # ---------------------------------------------------------
    index.add(vectors)

    # ---------------------------------------------------------
    # 5. Save the updated FAISS index
    # ---------------------------------------------------------
    faiss.write_index(index, str(index_path))

    # ---------------------------------------------------------
    # 6. Append metadata for the newly indexed chunks
    # ---------------------------------------------------------
    #
    # The order here MUST match the order in which vectors were
    # added to FAISS.
    # ---------------------------------------------------------
    existing_metadata.extend(
        c.model_dump()
        for c in chunks
    )

    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(
            existing_metadata,
            f,
            ensure_ascii=False,
            indent=2,
        )

    # ---------------------------------------------------------
    # 7. Report final index status
    # ---------------------------------------------------------
    size_mb = index_path.stat().st_size / (1024 ** 2)

    print(
        f"\n✅ Successfully updated HNSW index."
        f"\n   Added chunks : {len(chunks)}"
        f"\n   Total vectors: {index.ntotal}"
        f"\n   Dimensions   : {dim}"
        f"\n   HNSW M       : {m}"
        f"\n   Index size   : {size_mb:.2f} MB"
        f"\n   Index path   : {index_path}"
    )

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