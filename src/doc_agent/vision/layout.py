"""Stage 2 — layout detection using DocLayout-YOLO."""

from __future__ import annotations
import json
from pathlib import Path
from PIL import Image
from ..contracts import Page, Region

# DocLayout-YOLO labels -> fixed Region.kind contract.
_LABEL_MAP = {
    "title": "heading",
    "text": "text",
    "plain text": "text",
    "list": "text",
    "table": "table",
    "figure": "figure",
}

_model = None

REGION_DIR = Path("data/interim/regions")

def _get_model(cfg: dict):
    """Load the pretrained DocLayout-YOLO checkpoint once."""
    global _model

    if _model is not None:
        return _model
    from doclayout_yolo import YOLOv10
    model_path = (
        "/root/.cache/huggingface/hub/models--juliozhao--"
        "DocLayout-YOLO-DocStructBench/snapshots/"
        "8c3299a30b8ff29a1503c4431b035b93220f7b11/"
        "doclayout_yolo_docstructbench_imgsz1024.pt"
    )
    if not Path(model_path).exists():
        raise FileNotFoundError(
            f"DocLayout-YOLO checkpoint not found: {model_path}"
        )
    _model = YOLOv10(model_path)

    return _model


def _sort_reading_order(
    boxes: list[dict],
    page_width: int,
) -> list[dict]:
    """Sort regions approximately in multi-column reading order."""

    if not boxes:
        return boxes

    gap_px = 0.04 * page_width

    columns: list[list[dict]] = []

    for box in sorted(
        boxes,
        key=lambda b: (b["bbox"][0], b["bbox"][1]),
    ):
        x0 = box["bbox"][0]
        placed = False

        for column in columns:
            column_x0 = min(
                b["bbox"][0] for b in column
            )

            if abs(x0 - column_x0) <= gap_px * 3:
                column.append(box)
                placed = True
                break

        if not placed:
            columns.append([box])

    columns.sort(
        key=lambda column: min(
            b["bbox"][0] for b in column
        )
    )

    ordered: list[dict] = []

    for column in columns:
        column.sort(
            key=lambda b: b["bbox"][1]
        )
        ordered.extend(column)

    return ordered


def _save_regions(
    regions: list[Region],
    doc_id: str,
) -> None:
    """Save detected regions as JSON for reuse by later pipeline stages."""

    REGION_DIR.mkdir(parents=True, exist_ok=True)

    output_path = REGION_DIR / f"{doc_id}.json"

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(
            [region.model_dump() for region in regions],
            f,
            indent=2,
        )
    print(f"Saved {len(regions)} regions to {output_path}")


def detect(
    pages: list[Page],
    cfg: dict,
) -> list[Region]:
    """Detect layout regions and save them for later pipeline stages."""

    if not pages:
        return []

    model = _get_model(cfg)

    score_thr = (
        cfg.get("layout", {})
        .get("score_thr")
    )

    device = cfg.get("device", "cuda")
    regions: list[Region] = []
    for page in pages:
        image_path = Path(page.image_path)

        with Image.open(image_path) as image:
            image = image.convert("RGB")

            results = model.predict(
                image,
                conf=score_thr,
                device=device,
                verbose=False,
            )

            boxes: list[dict] = []

            for result in results:
                if result.boxes is None:
                    continue

                for box in result.boxes:
                    x0, y0, x1, y1 = map(
                        int,
                        box.xyxy[0].tolist(),
                    )

                    confidence = float(
                        box.conf[0].item()
                    )

                    class_id = int(
                        box.cls[0].item()
                    )

                    label = result.names[class_id]

                    kind = _LABEL_MAP.get(
                        label.lower(),
                        "text",
                    )

                    boxes.append(
                        {
                            "bbox": (x0, y0, x1, y1),
                            "kind": kind,
                            "score": confidence,
                        }
                    )

            ordered_boxes = _sort_reading_order(
                boxes,
                page_width=image.width,
            )

            for box in ordered_boxes:
                regions.append(
                    Region(
                        page_id=page.id,
                        bbox=box["bbox"],
                        kind=box["kind"],
                    )
                )

    # Save all regions belonging to this document.
    _save_regions(
        regions,
        pages[0].doc_id,
    )
    
    return regions



