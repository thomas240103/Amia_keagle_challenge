"""Original-image dimension helpers.

The competition train boxes are in original scan coordinate space. The PNG
files used for model input may be resized, commonly 1024x1024. Training boxes
therefore need to be scaled from original space into PNG space, while inference
boxes need to be scaled back for submission.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .columns import ImageSizeColumns, infer_image_size_columns
from .image_paths import image_id_variants


SizeMap = dict[str, tuple[float, float]]


def load_image_size_map(path: str | Path | None) -> tuple[SizeMap, ImageSizeColumns | None]:
    if path is None:
        return {}, None
    path = Path(path)
    if not path.exists():
        return {}, None
    df = pd.read_csv(path)
    columns = infer_image_size_columns(df)
    return build_image_size_map(df, columns), columns


def build_image_size_map(df: pd.DataFrame, columns: ImageSizeColumns) -> SizeMap:
    size_map: SizeMap = {}
    for _, row in df.iterrows():
        try:
            width = float(row[columns.width])
            height = float(row[columns.height])
        except Exception:
            continue
        if width <= 0 or height <= 0:
            continue
        for variant in image_id_variants(str(row[columns.image_id])):
            size_map[variant] = (width, height)
    return size_map


def lookup_original_size(image_id: str, size_map: SizeMap) -> tuple[float, float] | None:
    for variant in image_id_variants(str(image_id)):
        size = size_map.get(variant)
        if size is not None:
            return size
    return None


def scale_boxes_between_sizes(
    boxes: np.ndarray,
    from_width: float,
    from_height: float,
    to_width: float,
    to_height: float,
) -> np.ndarray:
    boxes = np.asarray(boxes, dtype=float).copy()
    if boxes.size == 0:
        return boxes.reshape(-1, 4)
    if from_width <= 0 or from_height <= 0:
        return boxes
    scale_x = float(to_width) / float(from_width)
    scale_y = float(to_height) / float(from_height)
    boxes[:, [0, 2]] *= scale_x
    boxes[:, [1, 3]] *= scale_y
    return boxes


def scale_original_boxes_to_png(
    boxes: np.ndarray,
    image_id: str,
    size_map: SizeMap,
    png_width: float,
    png_height: float,
) -> tuple[np.ndarray, bool]:
    original_size = lookup_original_size(image_id, size_map)
    if original_size is None:
        return np.asarray(boxes, dtype=float), False
    original_width, original_height = original_size
    return (
        scale_boxes_between_sizes(boxes, original_width, original_height, png_width, png_height),
        not _same_size(original_width, original_height, png_width, png_height),
    )


def scale_png_boxes_to_original(
    boxes: np.ndarray,
    image_id: str,
    size_map: SizeMap,
    png_width: float,
    png_height: float,
) -> tuple[np.ndarray, bool]:
    original_size = lookup_original_size(image_id, size_map)
    if original_size is None:
        return np.asarray(boxes, dtype=float), False
    original_width, original_height = original_size
    return (
        scale_boxes_between_sizes(boxes, png_width, png_height, original_width, original_height),
        not _same_size(original_width, original_height, png_width, png_height),
    )


def _same_size(width_a: float, height_a: float, width_b: float, height_b: float) -> bool:
    return abs(float(width_a) - float(width_b)) < 1e-3 and abs(float(height_a) - float(height_b)) < 1e-3
