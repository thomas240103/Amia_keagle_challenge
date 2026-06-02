"""Faster R-CNN prediction helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from tqdm.auto import tqdm

from src.utils.boxes import classwise_nms, clip_boxes_to_image
from src.data.image_sizes import SizeMap, scale_png_boxes_to_original


def image_collate(batch):
    images, metas = zip(*batch)
    return list(images), list(metas)


@torch.no_grad()
def predict_dataset(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    conf_threshold: float,
    nms_iou: float,
    max_det_per_image: int,
    original_size_map: SizeMap | None = None,
) -> pd.DataFrame:
    model.eval()
    records = []

    for images, metas in tqdm(loader, desc="predict", leave=False):
        images = [image.to(device) for image in images]
        outputs = model(images)

        for output, meta in zip(outputs, metas):
            image_id = str(meta["image_id"])
            png_width = float(meta.get("width", 0))
            png_height = float(meta.get("height", 0))
            boxes = output["boxes"].detach()
            scores = output["scores"].detach()
            labels = output["labels"].detach()

            keep = (scores >= float(conf_threshold)) & (labels >= 1) & (labels <= 14)
            boxes = boxes[keep]
            scores = scores[keep]
            labels = labels[keep]

            if boxes.numel() > 0:
                keep_indices = classwise_nms(boxes, scores, labels, float(nms_iou))
                keep_indices = keep_indices[: int(max_det_per_image)]
                boxes = boxes[keep_indices].cpu()
                scores = scores[keep_indices].cpu()
                labels = labels[keep_indices].cpu()
                if original_size_map and png_width > 0 and png_height > 0:
                    boxes_np, _ = scale_png_boxes_to_original(
                        boxes.numpy(),
                        image_id=image_id,
                        size_map=original_size_map,
                        png_width=png_width,
                        png_height=png_height,
                    )
                    boxes_np = _clip_to_original_size(boxes_np, image_id, original_size_map)
                    boxes = torch.as_tensor(boxes_np, dtype=torch.float32)

            for box, score, label in zip(boxes, scores, labels):
                x1, y1, x2, y2 = [float(v) for v in box.tolist()]
                records.append(
                    {
                        "image_id": image_id,
                        "class_id": int(label.item()) - 1,
                        "confidence": float(score.item()),
                        "xmin": x1,
                        "ymin": y1,
                        "xmax": x2,
                        "ymax": y2,
                    }
                )

    columns = ["image_id", "class_id", "confidence", "xmin", "ymin", "xmax", "ymax"]
    return pd.DataFrame.from_records(records, columns=columns)


def save_predictions(df: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def _clip_to_original_size(boxes_np, image_id: str, original_size_map: SizeMap):
    from src.data.image_sizes import lookup_original_size

    original_size = lookup_original_size(image_id, original_size_map)
    if original_size is None:
        return boxes_np
    width, height = original_size
    return clip_boxes_to_image(boxes_np, int(round(width)), int(round(height)))
