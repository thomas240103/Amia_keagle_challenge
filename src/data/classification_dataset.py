"""Datasets for optional V2 global and crop classifiers."""

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


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class GlobalCXRDataset(Dataset):
    """Whole-image multi-label classifier dataset."""

    def __init__(
        self,
        train_df: pd.DataFrame,
        image_ids: list[str],
        image_index: dict[str, Path],
        columns: DetectionColumns,
        image_size: int,
        no_finding_class_id: int = 14,
    ) -> None:
        self.image_ids = [str(v) for v in image_ids]
        self.image_index = image_index
        self.image_size = int(image_size)
        self.label_map = build_multilabel_map(train_df, columns, no_finding_class_id)

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int):
        image_id = self.image_ids[idx]
        path = resolve_image_path(image_id, self.image_index)
        if path is None:
            raise FileNotFoundError(f"Could not resolve image path for image_id={image_id}")
        image = Image.open(path).convert("RGB").resize((self.image_size, self.image_size), Image.BILINEAR)
        image_tensor = normalize_tensor(F.to_tensor(image))
        labels = lookup_multilabel(image_id, self.label_map)
        return image_tensor, torch.as_tensor(labels, dtype=torch.float32), image_id


class CropCXRDataset(Dataset):
    """Ground-truth crop classifier dataset."""

    def __init__(
        self,
        train_df: pd.DataFrame,
        allowed_image_ids: list[str],
        image_index: dict[str, Path],
        columns: DetectionColumns,
        original_size_map: SizeMap,
        image_size: int,
        crop_margin: float,
        no_finding_class_id: int = 14,
    ) -> None:
        self.image_index = image_index
        self.original_size_map = original_size_map
        self.image_size = int(image_size)
        self.crop_margin = float(crop_margin)
        self.examples = build_crop_examples(train_df, allowed_image_ids, columns, no_finding_class_id)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int):
        example = self.examples[idx]
        image_id = example["image_id"]
        path = resolve_image_path(image_id, self.image_index)
        if path is None:
            raise FileNotFoundError(f"Could not resolve image path for image_id={image_id}")
        image = Image.open(path).convert("RGB")
        width, height = image.size
        box, _ = scale_original_boxes_to_png(
            np.asarray([example["box"]], dtype=float),
            image_id=image_id,
            size_map=self.original_size_map,
            png_width=width,
            png_height=height,
        )
        box = clip_boxes_to_image(expand_box(box[0], self.crop_margin), width, height)[0]
        crop = image.crop(tuple(float(v) for v in box)).resize((self.image_size, self.image_size), Image.BILINEAR)
        image_tensor = normalize_tensor(F.to_tensor(crop))
        label = torch.tensor(int(example["class_id"]), dtype=torch.long)
        return image_tensor, label, image_id


def build_multilabel_map(
    train_df: pd.DataFrame,
    columns: DetectionColumns,
    no_finding_class_id: int = 14,
) -> dict[str, np.ndarray]:
    label_map: dict[str, np.ndarray] = {}
    rows = object_rows(train_df, columns, no_finding_class_id)
    for _, row in rows.iterrows():
        try:
            class_id = int(row[columns.class_id])
        except Exception:
            continue
        if class_id < 0 or class_id > 13:
            continue
        image_id = str(row[columns.image_id])
        for variant in image_id_variants(image_id):
            label_map.setdefault(variant, np.zeros((14,), dtype=np.float32))[class_id] = 1.0
    return label_map


def lookup_multilabel(image_id: str, label_map: dict[str, np.ndarray]) -> np.ndarray:
    for variant in image_id_variants(image_id):
        labels = label_map.get(variant)
        if labels is not None:
            return labels.copy()
    return np.zeros((14,), dtype=np.float32)


def build_crop_examples(
    train_df: pd.DataFrame,
    allowed_image_ids: list[str],
    columns: DetectionColumns,
    no_finding_class_id: int = 14,
) -> list[dict]:
    allowed = {variant for image_id in allowed_image_ids for variant in image_id_variants(image_id)}
    rows = object_rows(train_df, columns, no_finding_class_id)
    boxes, labels = extract_boxes_and_labels(rows, columns)
    valid = valid_box_mask(boxes) & np.isfinite(labels) & (labels >= 0) & (labels <= 13)
    rows = rows.loc[valid].reset_index(drop=True)
    boxes = boxes[valid]
    labels = labels[valid].astype(int)

    examples = []
    for (_, row), box, label in zip(rows.iterrows(), boxes, labels):
        image_id = str(row[columns.image_id])
        if not (image_id_variants(image_id) & allowed):
            continue
        examples.append({"image_id": image_id, "box": [float(v) for v in box], "class_id": int(label)})
    return examples


def expand_box(box: np.ndarray, margin: float) -> np.ndarray:
    x1, y1, x2, y2 = [float(v) for v in box]
    width = x2 - x1
    height = y2 - y1
    pad_x = width * float(margin)
    pad_y = height * float(margin)
    return np.asarray([[x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y]], dtype=float)


def normalize_tensor(tensor: torch.Tensor) -> torch.Tensor:
    return F.normalize(tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD)
