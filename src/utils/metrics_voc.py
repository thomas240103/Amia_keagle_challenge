"""VOC-style detection metrics used as a validation proxy."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import numpy as np

from .boxes import box_iou_matrix


def voc_ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
    if recalls.size == 0:
        return 0.0
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    changing = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[changing + 1] - mrec[changing]) * mpre[changing + 1]))


def voc_map(
    ground_truth: Iterable[dict],
    predictions: Iterable[dict],
    class_ids: Iterable[int] = range(14),
    iou_threshold: float = 0.4,
) -> dict:
    gt_by_class_image = defaultdict(lambda: defaultdict(list))
    pred_by_class = defaultdict(list)

    for row in ground_truth:
        gt_by_class_image[int(row["class_id"])][str(row["image_id"])].append(row["box"])

    for row in predictions:
        pred_by_class[int(row["class_id"])].append(row)

    per_class_ap = {}
    for class_id in class_ids:
        gt_for_class = gt_by_class_image[class_id]
        npos = sum(len(v) for v in gt_for_class.values())
        if npos == 0:
            per_class_ap[class_id] = None
            continue

        detected = {image_id: np.zeros(len(boxes), dtype=bool) for image_id, boxes in gt_for_class.items()}
        preds = sorted(pred_by_class[class_id], key=lambda x: float(x["score"]), reverse=True)
        tp = np.zeros(len(preds), dtype=float)
        fp = np.zeros(len(preds), dtype=float)

        for i, pred in enumerate(preds):
            image_id = str(pred["image_id"])
            gt_boxes = np.asarray(gt_for_class.get(image_id, []), dtype=float)
            if gt_boxes.size == 0:
                fp[i] = 1.0
                continue

            ious = box_iou_matrix(np.asarray([pred["box"]], dtype=float), gt_boxes)[0]
            best_idx = int(np.argmax(ious))
            best_iou = float(ious[best_idx])
            if best_iou >= iou_threshold and not detected[image_id][best_idx]:
                tp[i] = 1.0
                detected[image_id][best_idx] = True
            else:
                fp[i] = 1.0

        fp_cum = np.cumsum(fp)
        tp_cum = np.cumsum(tp)
        recalls = tp_cum / max(float(npos), 1.0)
        precisions = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)
        per_class_ap[class_id] = voc_ap(recalls, precisions)

    valid_aps = [ap for ap in per_class_ap.values() if ap is not None]
    return {
        "map": float(np.mean(valid_aps)) if valid_aps else 0.0,
        "per_class_ap": per_class_ap,
        "iou_threshold": iou_threshold,
    }
