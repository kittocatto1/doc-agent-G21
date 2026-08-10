"""Stage 1 — load scanned page-images."""

from __future__ import annotations
from pathlib import Path
from pdf2image import convert_from_path
from ..contracts import Page

RAW_DIR = Path("data/raw")
PAGE_DIR = Path("data/raw/pages")


def load_pages(cfg: dict) -> list[Page]:
    """Read data/raw/ and return scanned pages as Page objects."""

    pdf_files = sorted(RAW_DIR.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(
            f"No PDF files found in {RAW_DIR.resolve()}"
        )

    if len(pdf_files) > 1:
        raise ValueError(
            f"Expected one PDF in {RAW_DIR}, "
            f"but found {len(pdf_files)}. "
            "Process one issue at a time."
        )

    pdf_path = pdf_files[0]
    doc_id = pdf_path.stem

    output_dir = PAGE_DIR / doc_id
    output_dir.mkdir(parents=True, exist_ok=True)

    pages: list[Page] = []
    page_number = 1

    while True:
        image_path = output_dir / f"page_{page_number:04d}.png"

        # Reuse an already-converted page if it exists.
        if image_path.exists():
            pages.append(
                Page(
                    id=f"{doc_id}_p{page_number:04d}",
                    image_path=str(image_path),
                    doc_id=doc_id,
                )
            )
            page_number += 1
            continue

        try:
            images = convert_from_path(
                str(pdf_path),
                dpi=150,
                first_page=page_number,
                last_page=page_number,
                fmt="png",
            )
        except Exception as exc:
            if page_number == 1:
                raise RuntimeError(
                    f"Failed to convert the first page of {pdf_path}"
                ) from exc
            break

        # No image means we have reached the end of the PDF.
        if not images:
            break

        images[0].save(image_path, "PNG")

        pages.append(
            Page(
                id=f"{doc_id}_p{page_number:04d}",
                image_path=str(image_path),
                doc_id=doc_id,
            )
        )

        page_number += 1

    return pages