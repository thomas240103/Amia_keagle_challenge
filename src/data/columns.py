"""Column inference and annotation normalization."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from src.utils.boxes import valid_box_mask, xywh_to_xyxy


@dataclass(frozen=True)
class DetectionColumns:
    image_id: str
    class_id: str
    xmin: str
    ymin: str
    xmax: str | None = None
    ymax: str | None = None
    width: str | None = None
    height: str | None = None

    @property
    def uses_xywh(self) -> bool:
        return self.width is not None and self.height is not None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ImageSizeColumns:
    image_id: str
    width: str
    height: str

    def to_dict(self) -> dict:
        return asdict(self)


def normalize_name(name: str) -> str:
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def _find_column(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    normalized = {normalize_name(col): col for col in columns}
    for candidate in candidates:
        key = normalize_name(candidate)
        if key in normalized:
            return normalized[key]
    return None


def infer_detection_columns(df: pd.DataFrame) -> DetectionColumns:
    columns = list(df.columns)

    image_col = _find_column(
        columns,
        [
            "image_id",
            "imageid",
            "id",
            "image",
            "filename",
            "file_name",
            "image_name",
            "image_path",
            "path",
            "study_id",
        ],
    )
    class_col = _find_column(
        columns,
        [
            "class_id",
            "classid",
            "class",
            "label",
            "target",
            "category_id",
            "category",
        ],
    )
    xmin = _find_column(columns, ["xmin", "x_min", "x1", "left"])
    ymin = _find_column(columns, ["ymin", "y_min", "y1", "top"])
    xmax = _find_column(columns, ["xmax", "x_max", "x2", "right"])
    ymax = _find_column(columns, ["ymax", "y_max", "y2", "bottom"])
    x = _find_column(columns, ["x", "bbox_x"])
    y = _find_column(columns, ["y", "bbox_y"])
    width = _find_column(columns, ["width", "w", "bbox_width"])
    height = _find_column(columns, ["height", "h", "bbox_height"])

    if image_col is None:
        raise ValueError(f"Could not infer image id column from: {columns}")
    if class_col is None:
        raise ValueError(f"Could not infer class id column from: {columns}")
    if xmin and ymin and xmax and ymax:
        return DetectionColumns(image_col, class_col, xmin, ymin, xmax=xmax, ymax=ymax)
    if x and y and width and height:
        return DetectionColumns(image_col, class_col, x, y, width=width, height=height)

    raise ValueError(
        "Could not infer bounding box columns. Expected xmin/ymin/xmax/ymax or x/y/width/height."
    )


def infer_image_id_column(df: pd.DataFrame) -> str:
    image_col = _find_column(
        list(df.columns),
        [
            "image_id",
            "imageid",
            "id",
            "image",
            "filename",
            "file_name",
            "image_name",
            "image_path",
            "path",
            "study_id",
        ],
    )
    if image_col is None:
        raise ValueError(f"Could not infer image id column from: {list(df.columns)}")
    return image_col


def infer_image_size_columns(df: pd.DataFrame) -> ImageSizeColumns:
    columns = list(df.columns)
    image_col = infer_image_id_column(df)
    width = _find_column(columns, ["dim1", "width", "w", "image_width", "img_width", "cols", "columns"])
    height = _find_column(columns, ["dim0", "height", "h", "image_height", "img_height", "rows"])
    if width is None or height is None:
        raise ValueError(f"Could not infer image size columns from: {columns}")
    return ImageSizeColumns(image_col, width, height)


def extract_image_ids(df: pd.DataFrame, image_col: str) -> list[str]:
    return [str(v) for v in df[image_col].dropna().astype(str).unique().tolist()]


def extract_boxes_and_labels(df: pd.DataFrame, columns: DetectionColumns) -> tuple[np.ndarray, np.ndarray]:
    if len(df) == 0:
        return np.zeros((0, 4), dtype=float), np.zeros((0,), dtype=int)

    if columns.uses_xywh:
        boxes = df[[columns.xmin, columns.ymin, columns.width, columns.height]].apply(
            pd.to_numeric, errors="coerce"
        ).to_numpy(dtype=float)
        boxes = xywh_to_xyxy(boxes)
    else:
        boxes = df[[columns.xmin, columns.ymin, columns.xmax, columns.ymax]].apply(
            pd.to_numeric, errors="coerce"
        ).to_numpy(dtype=float)

    labels = pd.to_numeric(df[columns.class_id], errors="coerce").to_numpy()
    return boxes, labels


def object_rows(df: pd.DataFrame, columns: DetectionColumns, no_finding_class_id: int = 14) -> pd.DataFrame:
    labels = pd.to_numeric(df[columns.class_id], errors="coerce")
    return df[(labels.notna()) & (labels.astype(int) != int(no_finding_class_id))].copy()


def validate_annotation_rows(
    df: pd.DataFrame,
    columns: DetectionColumns,
    no_finding_class_id: int = 14,
) -> dict:
    obj = object_rows(df, columns, no_finding_class_id)
    boxes, labels = extract_boxes_and_labels(obj, columns)
    valid_box = valid_box_mask(boxes)
    finite_label = np.isfinite(labels)
    integer_label = np.zeros_like(finite_label, dtype=bool)
    integer_label[finite_label] = labels[finite_label] == labels[finite_label].astype(int)
    valid_label = finite_label & integer_label & (labels >= 0) & (labels <= 13)
    valid = valid_box & valid_label
    no_finding_rows = int((pd.to_numeric(df[columns.class_id], errors="coerce") == no_finding_class_id).sum())
    return {
        "rows": int(len(df)),
        "object_rows": int(len(obj)),
        "no_finding_rows": no_finding_rows,
        "valid_object_rows": int(valid.sum()),
        "invalid_object_rows": int((~valid).sum()),
    }
