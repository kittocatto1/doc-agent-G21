"""Data — corpus versioning (which corpus version -> which result)"""
from __future__ import annotations
import hashlib
import json
import os
from datetime import datetime, timezone


def snapshot(corpus_dir: str) -> str:
    """Hash + record a corpus version id."""
    file_records = []
    for root, _, files in os.walk(corpus_dir):
        for fname in sorted(files):
            fpath = os.path.join(root, fname)
            size = os.path.getsize(fpath)
            with open(fpath, "rb") as f:
                content_hash = hashlib.sha256(f.read()).hexdigest()[:16]
            file_records.append(f"{os.path.relpath(fpath, corpus_dir)}:{size}:{content_hash}")

    file_records.sort()
    combined = "\n".join(file_records)
    version_id = hashlib.sha256(combined.encode()).hexdigest()[:12]

    record = {
        "version_id": version_id,
        "corpus_dir": corpus_dir,
        "n_files": len(file_records),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    os.makedirs("data/versions", exist_ok=True)
    out_path = f"data/versions/{version_id}.json"
    with open(out_path, "w") as f:
        json.dump(record, f, indent=2)

    print(f"Corpus snapshot saved: {version_id} ({len(file_records)} files) → {out_path}")
    return version_id