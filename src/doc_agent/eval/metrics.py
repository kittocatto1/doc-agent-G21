"""Stage 9 — metrics"""
from __future__ import annotations
from ..contracts import *  # noqa

def ocr_f1(pred: str, gold: str) -> float:
    """Word-level F1 between OCR output and ground truth."""
    pred_words = pred.lower().split()
    gold_words = gold.lower().split()

    if not pred_words and not gold_words:
        return 1.0
    if not pred_words or not gold_words:
        return 0.0

    from collections import Counter
    pred_counts = Counter(pred_words)
    gold_counts = Counter(gold_words)

    overlap = sum((pred_counts & gold_counts).values())  # multiset intersection

    precision = overlap / len(pred_words)
    recall = overlap / len(gold_words)

    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)
def recall_at_k(retrieved: list, gold: list, k: int) -> float: raise NotImplementedError
def groundedness(answer: Answer) -> float: raise NotImplementedError  # no-hallucination
def citation_accuracy(answer: Answer) -> float: raise NotImplementedError
def ece(confidences, correct) -> float: raise NotImplementedError     # calibration
def subgroup_gap(scores_by_group: dict) -> float: raise NotImplementedError  # fairness

