"""Bounding box helpers."""

from __future__ import annotations

import numpy as np


def xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    boxes = boxes.astype(float, copy=True)
    boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
    boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
    return boxes


def clip_boxes_to_image(boxes: np.ndarray, width: int, height: int) -> np.ndarray:
    boxes = boxes.astype(float, copy=True)
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, max(width - 1, 0))
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, max(height - 1, 0))
    return boxes


def valid_box_mask(boxes: np.ndarray, min_size: float = 1.0) -> np.ndarray:
    boxes = np.asarray(boxes, dtype=float)
    if boxes.size == 0:
        return np.zeros((0,), dtype=bool)
    finite = np.isfinite(boxes).all(axis=1)
    positive = (boxes[:, 2] - boxes[:, 0] >= min_size) & (boxes[:, 3] - boxes[:, 1] >= min_size)
    non_negative = (boxes[:, 0] >= 0) & (boxes[:, 1] >= 0)
    return finite & positive & non_negative


def box_iou_matrix(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    boxes1 = np.asarray(boxes1, dtype=float)
    boxes2 = np.asarray(boxes2, dtype=float)
    if boxes1.size == 0 or boxes2.size == 0:
        return np.zeros((len(boxes1), len(boxes2)), dtype=float)

    x1 = np.maximum(boxes1[:, None, 0], boxes2[None, :, 0])
    y1 = np.maximum(boxes1[:, None, 1], boxes2[None, :, 1])
    x2 = np.minimum(boxes1[:, None, 2], boxes2[None, :, 2])
    y2 = np.minimum(boxes1[:, None, 3], boxes2[None, :, 3])

    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area1 = np.maximum(0, boxes1[:, 2] - boxes1[:, 0]) * np.maximum(0, boxes1[:, 3] - boxes1[:, 1])
    area2 = np.maximum(0, boxes2[:, 2] - boxes2[:, 0]) * np.maximum(0, boxes2[:, 3] - boxes2[:, 1])
    union = area1[:, None] + area2[None, :] - inter
    return np.divide(inter, union, out=np.zeros_like(inter, dtype=float), where=union > 0)


def classwise_nms(boxes, scores, labels, iou_threshold: float):
    """Run class-wise NMS on torch tensors and return kept indices."""
    import torch

    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.long, device=boxes.device)

    try:
        from torchvision.ops import nms
    except Exception:
        nms = None

    kept = []
    for class_id in labels.unique():
        class_indices = torch.where(labels == class_id)[0]
        class_boxes = boxes[class_indices]
        class_scores = scores[class_indices]
        if nms is not None:
            class_keep = class_indices[nms(class_boxes, class_scores, iou_threshold)]
        else:
            class_keep = _numpy_nms(class_indices, class_boxes, class_scores, iou_threshold)
        kept.append(class_keep)

    if not kept:
        return torch.empty((0,), dtype=torch.long, device=boxes.device)
    keep = torch.cat(kept)
    return keep[torch.argsort(scores[keep], descending=True)]


def _numpy_nms(indices, boxes, scores, iou_threshold: float):
    import torch

    boxes_np = boxes.detach().cpu().numpy()
    scores_np = scores.detach().cpu().numpy()
    order = scores_np.argsort()[::-1]
    keep_local = []

    while order.size > 0:
        i = order[0]
        keep_local.append(i)
        if order.size == 1:
            break
        ious = box_iou_matrix(boxes_np[[i]], boxes_np[order[1:]])[0]
        order = order[1:][ious <= iou_threshold]

    return indices[torch.as_tensor(keep_local, dtype=torch.long, device=indices.device)]
