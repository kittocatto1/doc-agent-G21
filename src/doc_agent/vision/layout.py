"""Stage 2 — layout detection / segmentation"""
from __future__ import annotations
from PIL import Image
from ..contracts import Page, Region

# !pip install layoutparser[layoutmodels] -q

# Maps the pretrained model's own label names to the 4 kinds our contract allows.
_LABEL_MAP = {
    "Text": "text",
    "Title": "heading",
    "List": "text",
    "Table": "table",
    "Figure": "figure",
}

_model = None  # lazy-loaded once, reused across all pages in this run


def _get_model(cfg: dict):
    global _model
    if _model is not None:
        return _model
    import layoutparser as lp
    score_thr = cfg.get("layout", {}).get("score_thr", 0.5)
    _model = lp.Detectron2LayoutModel(
        config_path="lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config",
        extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", score_thr],
        label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"},
    )
    return _model


def _sort_reading_order(boxes: list[dict], page_width: int) -> list[dict]:
    """Sort boxes into multi-column reading order (E2): left column top-to-bottom,
    then next column — NOT naive top-to-bottom across the whole page, which would
    scramble BYTE's 2-3 column layout."""
    if not boxes:
        return boxes

    gap_px = 0.04 * page_width  # tolerance for "same column" grouping

    columns: list[list[dict]] = []
    for box in sorted(boxes, key=lambda b: b["bbox"][0]):
        x0 = box["bbox"][0]
        placed = False
        for col in columns:
            col_x0 = min(b["bbox"][0] for b in col)
            if abs(x0 - col_x0) <= gap_px * 3:
                col.append(box)
                placed = True
                break
        if not placed:
            columns.append([box])

    columns.sort(key=lambda col: min(b["bbox"][0] for b in col))  # left-to-right

    ordered: list[dict] = []
    for col in columns:
        col.sort(key=lambda b: b["bbox"][1])  # top-to-bottom within column
        ordered.extend(col)
    return ordered


def detect(pages: list[Page], cfg: dict) -> list[Region]:
    """Detect text/table/figure/heading regions, in correct multi-column reading order."""
    model = _get_model(cfg)
    regions: list[Region] = []

    for page in pages:
        image = Image.open(page.image_path).convert("RGB")
        layout = model.detect(image)

        boxes = []
        for block in layout:
            kind = _LABEL_MAP.get(block.type, "text")
            x0, y0, x1, y1 = map(int, block.coordinates)
            boxes.append({"bbox": (x0, y0, x1, y1), "kind": kind})

        ordered_boxes = _sort_reading_order(boxes, page_width=image.width)

        for box in ordered_boxes:
            regions.append(Region(page_id=page.id, bbox=box["bbox"], kind=box["kind"]))

    return regions