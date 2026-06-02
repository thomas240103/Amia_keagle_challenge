"""Torch datasets for Faster R-CNN training and inference."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as F

from src.utils.boxes import clip_boxes_to_image, valid_box_mask

from .columns import DetectionColumns, extract_boxes_and_labels, object_rows
from .image_paths import image_id_variants, resolve_image_path
from .image_sizes import SizeMap, scale_original_boxes_to_png


class CXRDetectionDataset(Dataset):
    def __init__(
        self,
        annotations: pd.DataFrame,
        image_ids: list[str],
        image_index: dict[str, Path],
        columns: DetectionColumns,
        no_finding_class_id: int = 14,
        original_size_map: SizeMap | None = None,
    ) -> None:
        self.annotations = annotations.copy()
        self.image_ids = [str(v) for v in image_ids]
        self.image_index = image_index
        self.columns = columns
        self.no_finding_class_id = int(no_finding_class_id)
        self.original_size_map = original_size_map or {}
        self.rows_by_image = {
            variant: group.copy()
            for image_id, group in self.annotations.groupby(self.columns.image_id, sort=False)
            for variant in image_id_variants(str(image_id))
        }

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int):
        image_id = self.image_ids[idx]
        path = resolve_image_path(image_id, self.image_index)
        if path is None:
            raise FileNotFoundError(f"Could not resolve image path for image_id={image_id}")

        image = Image.open(path).convert("RGB")
        width, height = image.size
        rows = self._lookup_rows(image_id)

        if rows is None or len(rows) == 0:
            boxes = np.zeros((0, 4), dtype=np.float32)
            labels = np.zeros((0,), dtype=np.int64)
        else:
            rows = object_rows(rows, self.columns, self.no_finding_class_id)
            boxes, labels = extract_boxes_and_labels(rows, self.columns)
            boxes, _ = scale_original_boxes_to_png(
                boxes,
                image_id=image_id,
                size_map=self.original_size_map,
                png_width=width,
                png_height=height,
            )
            boxes = clip_boxes_to_image(boxes, width, height)
            keep = valid_box_mask(boxes) & (labels >= 0) & (labels <= 13)
            boxes = boxes[keep].astype(np.float32)
            labels = labels[keep].astype(np.int64) + 1

        image_tensor = F.to_tensor(image)
        boxes_tensor = torch.as_tensor(boxes, dtype=torch.float32)
        labels_tensor = torch.as_tensor(labels, dtype=torch.int64)
        area = (
            (boxes_tensor[:, 2] - boxes_tensor[:, 0]) * (boxes_tensor[:, 3] - boxes_tensor[:, 1])
            if boxes_tensor.numel()
            else torch.zeros((0,), dtype=torch.float32)
        )
        target = {
            "boxes": boxes_tensor.reshape(-1, 4),
            "labels": labels_tensor,
            "image_id": torch.tensor([idx], dtype=torch.int64),
            "area": area,
            "iscrowd": torch.zeros((len(labels_tensor),), dtype=torch.int64),
        }
        return image_tensor, target

    def _lookup_rows(self, image_id: str) -> pd.DataFrame | None:
        for variant in image_id_variants(image_id):
            rows = self.rows_by_image.get(variant)
            if rows is not None:
                return rows
        return None


class CXRImageDataset(Dataset):
    def __init__(self, image_ids: list[str], image_index: dict[str, Path]) -> None:
        self.image_ids = [str(v) for v in image_ids]
        self.image_index = image_index

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int):
        image_id = self.image_ids[idx]
        path = resolve_image_path(image_id, self.image_index)
        if path is None:
            raise FileNotFoundError(f"Could not resolve image path for image_id={image_id}")
        image = Image.open(path).convert("RGB")
        width, height = image.size
        return F.to_tensor(image), {"image_id": image_id, "path": str(path), "width": width, "height": height}
