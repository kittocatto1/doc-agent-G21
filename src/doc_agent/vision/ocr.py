"""Stage 3 — OCR using Qwen2-VL-2B-Instruct."""
from __future__ import annotations
from pathlib import Path
import torch
import json 
from PIL import Image
from ..contracts import Chunk, Region

OCR_DIR = Path("data/interim/ocr")

_model = None
_processor = None

def _get_model(cfg: dict):
    """Load the configured VLM once and reuse it."""

    global _model, _processor

    if _model is not None and _processor is not None:
        return _model, _processor

    from transformers import (
        AutoProcessor,
        Qwen2VLForConditionalGeneration,
    )

    model_name = cfg["ocr"]["model"]

    print(f"Loading OCR model: {model_name}")

    _processor = AutoProcessor.from_pretrained(model_name)

    _model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto",
    )

    _model.eval()

    return _model, _processor


def _find_page_image(page_id: str) -> Path:
    """Find the preprocessed page image belonging to a Region."""

    if "_p" not in page_id:
        raise ValueError(
            f"Unexpected page_id format: {page_id}"
        )

    doc_id, page_number = page_id.rsplit("_p", 1)

    filename = f"page_{page_number}.png"

    search_dirs = [
        Path("data/interim/preprocessed"),
        Path("data/interim/pages")
    ]

    for directory in search_dirs:
        candidate = directory / doc_id / filename

        if candidate.exists():
            return candidate

    # Fallback for slightly different directory structures.
    for directory in search_dirs:
        if directory.exists():
            matches = list(directory.rglob(filename))

            for path in matches:
                if doc_id in str(path):
                    return path

    raise FileNotFoundError(
        f"Could not find image for page_id={page_id}"
    )


def _crop_region(region: Region) -> Image.Image:
    """Load a page and crop the detected region."""

    image_path = _find_page_image(region.page_id)

    with Image.open(image_path) as image:
        image = image.convert("RGB")

        x0, y0, x1, y1 = region.bbox

        # Keep bbox inside image boundaries.
        x0 = max(0, min(x0, image.width))
        y0 = max(0, min(y0, image.height))
        x1 = max(0, min(x1, image.width))
        y1 = max(0, min(y1, image.height))

        if x1 <= x0 or y1 <= y0:
            raise ValueError(
                f"Invalid region bbox: {region.bbox}"
            )

        return image.crop((x0, y0, x1, y1))


def _prompt_for_region(kind: str) -> str:
    """Return an OCR instruction appropriate for the region type."""

    if kind == "heading":
        return (
            "Transcribe the heading exactly as shown. "
            "Return only the text. Do not explain or summarize."
        )

    if kind == "table":
        return (
            "Transcribe all readable text in this table. "
            "Preserve rows and columns as clearly as possible. "
            "Return only the transcription."
        )

    if kind == "figure":
        return (
            "Transcribe any readable text appearing in this figure, "
            "diagram, or caption. Return only the text. "
            "Do not describe the image."
        )

    return (
        "Transcribe all visible text exactly as shown. "
        "Preserve words, numbers, punctuation, and line breaks "
        "where possible. Do not summarize or explain."
    )


class Reader:
    """VLM-based OCR reader."""

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg["ocr"]
        self.model, self.processor = _get_model(cfg)

    def transcribe_region(self, region: Region) -> str:
        """OCR a single detected layout region."""

        crop = _crop_region(region)
        prompt = _prompt_for_region(region.kind)

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": crop,
                    },
                    {
                        "type": "text",
                        "text": prompt,
                    },
                ],
            }
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )

        # Move tensor inputs to the same device as the model.
        inputs = {
            key: value.to(self.model.device)
            if hasattr(value, "to")
            else value
            for key, value in inputs.items()
        }

        with torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.cfg.get(
                    "max_new_tokens",
                    512,
                ),
                do_sample=False,
            )

        input_length = inputs["input_ids"].shape[1]

        generated_ids = generated_ids[
            :, input_length:
        ]

        text = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        return text.strip()


def transcribe(
    regions: list[Region],
    cfg: dict,
) -> list[Chunk]:
    """Convert layout regions into OCR text chunks.
    Checkpoints incrementally to OCR_DIR/chunks.json and resumes from it
    on restart (skips regions whose chunk_id was already OCR'd) -- Kaggle
    sessions can die mid-run, this avoids losing completed work."""

    OCR_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OCR_DIR / "chunks.json"
    checkpoint_every = cfg["ocr"].get("checkpoint_every", 25)

    # Resume: load any chunks already OCR'd from a previous run.
    chunks: list[Chunk] = []
    already_done_ids: set[str] = set()
    if output_path.exists():
        with output_path.open("r", encoding="utf-8") as f:
            existing = json.load(f)
        chunks = [Chunk(**item) for item in existing]
        already_done_ids = {c.id for c in chunks}
        print(f"Resuming: {len(chunks)} chunks already OCR'd, skipping those.")

    reader = Reader(cfg)

    for index, region in enumerate(regions):
        doc_id = region.page_id.rsplit("_p", 1)[0]
        chunk_id = f"{region.page_id}_region_{index:04d}"

        if chunk_id in already_done_ids:
            continue  # already OCR'd in a previous (interrupted) run

        try:
            text = reader.transcribe_region(region)
        except Exception as exc:
            print(f"WARNING: OCR failed for region {index}/{len(regions)} ({chunk_id}): {exc}")
            continue

        if not text:
            continue

        chunks.append(Chunk(
            id=chunk_id, doc_id=doc_id, text=text,
            page_ids=[region.page_id], score=0.0,
        ))

        if (index + 1) % checkpoint_every == 0 or (index + 1) == len(regions):
            with output_path.open("w", encoding="utf-8") as f:
                json.dump([c.model_dump() for c in chunks], f, ensure_ascii=False, indent=2)
            print(f"[{index + 1}/{len(regions)}] checkpoint saved -- {len(chunks)} chunks total")

    return chunks