"""Stage 1 — deskew / denoise / binarize / augment

but for this corpus we considered —
    mild enhancement and denoising for scanned page-images."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageEnhance, ImageFilter

from ..contracts import Page


PROCESSED_DIR = Path("data/interim/preprocessed")


def run(pages: list[Page], cfg: dict) -> list[Page]:
    """Apply mild image enhancement while preserving page colors."""

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    processed_pages: list[Page] = []

    for page in pages:
        input_path = Path(page.image_path)

        output_dir = PROCESSED_DIR / page.doc_id
        output_dir.mkdir(parents=True, exist_ok=True)

        output_path = output_dir / input_path.name

        # Reuse an already-preprocessed page if it exists.
        if not output_path.exists():
            with Image.open(input_path) as image:
                # Preserve the original RGB/color information because
                # BYTE is a colorful magazine corpus containing figures,
                # advertisements, diagrams, headings, and other visual
                # layout information useful to downstream models.
                image = image.convert("RGB")

                # Mild denoising to reduce scan noise while preserving
                # fine text and code characters.
                image = image.filter(
                    ImageFilter.MedianFilter(size=3)
                )

                # Mild contrast enhancement to improve text/background
                # separation without aggressively altering the scan.
                image = ImageEnhance.Contrast(image).enhance(1.1)

                # Mild sharpening to improve fine character edges.
                image = ImageEnhance.Sharpness(image).enhance(1.1)

                image.save(output_path, "PNG")

        processed_pages.append(
            Page(
                id=page.id,
                image_path=str(output_path),
                doc_id=page.doc_id,
            )
        )

    return processed_pages