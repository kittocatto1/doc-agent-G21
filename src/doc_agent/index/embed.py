"""Stage 4 — embed chunks"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from ..contracts import Chunk

EMBED_DIR = Path("data/interim/embed")
_model = None


def _get_model(cfg: dict):
    global _model
    if _model is not None:
        return _model
    from sentence_transformers import SentenceTransformer
    model_name = cfg.get("embed", {}).get("model")
    _model = SentenceTransformer(model_name, trust_remote_code=True)
    return _model


def encode(chunks: list[Chunk], cfg: dict) -> np.ndarray:
    """Embed with cfg['embed']['model']."""
    model = _get_model(cfg)
    texts = [c.text for c in chunks]

    vectors = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    expected_dim = cfg.get("embed", {}).get("dim")
    if expected_dim and vectors.shape[1] != expected_dim:
        print(f"Warning: got dim {vectors.shape[1]}, config expected {expected_dim}")

    EMBED_DIR.mkdir(parents=True, exist_ok=True)
    np.save(EMBED_DIR / "vectors.npy", vectors)
    with open(EMBED_DIR / "chunk_ids.json", "w") as f:
        json.dump([c.id for c in chunks], f)
    print(f"Saved {vectors.shape[0]} vectors (dim={vectors.shape[1]}) to {EMBED_DIR}")

    return vectors