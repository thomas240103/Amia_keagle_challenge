"""Dimension and bounding-box audit for the competition dataset."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from PIL import Image

from src.utils.boxes import valid_box_mask
from src.utils.config import ensure_work_dir

from .columns import (
    DetectionColumns,
    extract_boxes_and_labels,
    infer_detection_columns,
    infer_image_id_column,
    infer_image_size_columns,
    object_rows,
)
from .discovery import discover_dataset, load_csv
from .image_paths import build_image_index, image_id_variants, image_ids_from_paths, resolve_image_path
from .image_sizes import build_image_size_map, lookup_original_size, scale_original_boxes_to_png


def run_dimension_audit(config: dict, max_images: int = 0) -> dict:
    work_dir = ensure_work_dir(config)
    no_finding_class_id = int(config["submission"]["no_finding_class_id"])
    scanner_cfg = config["scanner"]
    image_size = int(scanner_cfg["image_size"])
    max_size = int(scanner_cfg.get("max_size", round(image_size * 1.5)))

    discovery = discover_dataset(config["data_root"])
    if discovery.train_csv is None:
        raise RuntimeError("Cannot audit dimensions without train.csv")

    train_df = load_csv(discovery.train_csv)
    columns = infer_detection_columns(train_df)
    img_size_report = _audit_img_size_csv(discovery.img_size_csv)

    train_paths = discovery.train_image_paths or []
    test_paths = discovery.test_image_paths or []
    train_image_dims = _audit_pil_dimensions(train_paths, max_images=max_images)
    test_image_dims = _audit_pil_dimensions(test_paths, max_images=max_images)

    size_map = img_size_report.get("_size_map", {})
    png_size_map = _size_map_from_pil(train_image_dims.get("_records", []), test_image_dims.get("_records", []))
    if not size_map:
        size_map = png_size_map

    box_report = _audit_boxes(
        train_df=train_df,
        columns=columns,
        size_map=size_map,
        no_finding_class_id=no_finding_class_id,
        image_size=image_size,
        max_size=max_size,
        png_size_map=png_size_map,
    )

    train_folder_ids = image_ids_from_paths(train_paths) if train_paths else []
    train_csv_ids = [str(v) for v in train_df[columns.image_id].dropna().astype(str).unique().tolist()]
    alignment_report = {
        "train_folder_images": len(train_folder_ids),
        "train_csv_unique_ids": len(train_csv_ids),
        "train_csv_ids_missing_from_train_folder": _count_missing_ids(train_csv_ids, build_image_index(train_paths)),
    }

    test_report = {}
    if discovery.test_csv is not None:
        test_df = load_csv(discovery.test_csv)
        try:
            test_col = infer_image_id_column(test_df)
            test_csv_ids = [str(v) for v in test_df[test_col].dropna().astype(str).unique().tolist()]
            test_report = {
                "test_csv_path": str(discovery.test_csv),
                "test_csv_image_column": test_col,
                "test_csv_unique_ids": len(test_csv_ids),
                "test_csv_ids_missing_from_test_folder": _count_missing_ids(test_csv_ids, build_image_index(test_paths)),
            }
        except Exception as exc:
            test_report = {"test_csv_path": str(discovery.test_csv), "error": str(exc)}

    report = {
        "data_root": str(discovery.root),
        "train_csv": str(discovery.train_csv),
        "test_csv": str(discovery.test_csv) if discovery.test_csv else None,
        "img_size_csv": str(discovery.img_size_csv) if discovery.img_size_csv else None,
        "sample_submission": str(discovery.sample_submission) if discovery.sample_submission else None,
        "scanner_resize_config": {
            "image_size_min_side": image_size,
            "max_size_long_side": max_size,
            "meaning": "torchvision detection transform resizes the shortest side to image_size, capped by max_size on the longest side",
        },
        "img_size_csv_audit": _without_private(img_size_report),
        "train_image_file_audit": _without_private(train_image_dims),
        "test_image_file_audit": _without_private(test_image_dims),
        "csv_folder_alignment": alignment_report,
        "test_csv_audit": test_report,
        "train_box_audit": box_report,
        "recommendation": _recommend_resize(box_report, image_size, max_size),
    }

    output_path = work_dir / "lgcxr_dimension_audit.json"
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report["output_path"] = str(output_path)
    return report


def _audit_img_size_csv(path: Path | None) -> dict:
    if path is None:
        return {"present": False, "_size_map": {}}
    df = load_csv(path)
    columns = infer_image_size_columns(df)
    widths = pd.to_numeric(df[columns.width], errors="coerce").to_numpy(dtype=float)
    heights = pd.to_numeric(df[columns.height], errors="coerce").to_numpy(dtype=float)
    records = []
    size_map = build_image_size_map(df, columns)
    for _, row in df.iterrows():
        try:
            width = float(row[columns.width])
            height = float(row[columns.height])
        except Exception:
            continue
        if width <= 0 or height <= 0:
            continue
        image_id = str(row[columns.image_id])
        records.append({"image_id": image_id, "width": width, "height": height})
    return {
        "present": True,
        "path": str(path),
        "columns": columns.to_dict(),
        "rows": int(len(df)),
        "valid_rows": int(len(records)),
        "width": _numeric_summary(widths),
        "height": _numeric_summary(heights),
        "aspect_ratio": _numeric_summary(widths / np.maximum(heights, 1e-9)),
        "common_sizes": _common_sizes([(r["width"], r["height"]) for r in records]),
        "_size_map": size_map,
    }


def _audit_pil_dimensions(paths: list[Path], max_images: int = 0) -> dict:
    selected = paths if max_images <= 0 else paths[:max_images]
    records = []
    failures = []
    for path in selected:
        try:
            with Image.open(path) as image:
                width, height = image.size
            records.append({"image_id": path.stem, "path": str(path), "width": float(width), "height": float(height)})
        except Exception as exc:
            failures.append({"path": str(path), "error": str(exc)})
    widths = np.asarray([r["width"] for r in records], dtype=float)
    heights = np.asarray([r["height"] for r in records], dtype=float)
    return {
        "images_available": int(len(paths)),
        "images_checked": int(len(selected)),
        "valid_images": int(len(records)),
        "failures": failures[:10],
        "width": _numeric_summary(widths),
        "height": _numeric_summary(heights),
        "aspect_ratio": _numeric_summary(widths / np.maximum(heights, 1e-9)) if len(records) else {},
        "common_sizes": _common_sizes([(r["width"], r["height"]) for r in records]),
        "_records": records,
    }


def _audit_boxes(
    train_df: pd.DataFrame,
    columns: DetectionColumns,
    size_map: dict[str, tuple[float, float]],
    no_finding_class_id: int,
    image_size: int,
    max_size: int,
    png_size_map: dict[str, tuple[float, float]],
) -> dict:
    object_df = object_rows(train_df, columns, no_finding_class_id)
    boxes, labels = extract_boxes_and_labels(object_df, columns)
    valid = valid_box_mask(boxes) & np.isfinite(labels) & (labels >= 0) & (labels <= 13)
    boxes = boxes[valid]
    labels = labels[valid].astype(int)
    image_ids = object_df.loc[valid, columns.image_id].astype(str).tolist()

    widths = boxes[:, 2] - boxes[:, 0] if len(boxes) else np.asarray([])
    heights = boxes[:, 3] - boxes[:, 1] if len(boxes) else np.asarray([])
    areas = widths * heights
    class_counts = Counter(int(v) for v in labels.tolist())

    rel_widths = []
    rel_heights = []
    rel_areas = []
    png_box_widths = []
    png_box_heights = []
    resized_widths = []
    resized_heights = []
    missing_size = 0
    missing_png_size = 0
    outside_image = 0
    normalized_like = 0
    original_to_png_scaled = 0

    for image_id, box, bw, bh, area in zip(image_ids, boxes, widths, heights, areas):
        size = lookup_original_size(image_id, size_map)
        if size is None:
            missing_size += 1
            continue
        original_width, original_height = size
        x1, y1, x2, y2 = [float(v) for v in box]
        if max(x1, y1, x2, y2) <= 1.5:
            normalized_like += 1
        if x1 < 0 or y1 < 0 or x2 > original_width or y2 > original_height:
            outside_image += 1
        rel_widths.append(float(bw / max(original_width, 1e-9)))
        rel_heights.append(float(bh / max(original_height, 1e-9)))
        rel_areas.append(float(area / max(original_width * original_height, 1e-9)))

        png_size = lookup_original_size(image_id, png_size_map)
        if png_size is None:
            missing_png_size += 1
            continue
        png_width, png_height = png_size
        png_boxes, did_scale = scale_original_boxes_to_png(
            np.asarray([box], dtype=float),
            image_id=image_id,
            size_map=size_map,
            png_width=png_width,
            png_height=png_height,
        )
        if did_scale:
            original_to_png_scaled += 1
        png_bw = float(png_boxes[0, 2] - png_boxes[0, 0])
        png_bh = float(png_boxes[0, 3] - png_boxes[0, 1])
        png_box_widths.append(png_bw)
        png_box_heights.append(png_bh)
        scale = _torchvision_resize_scale(png_width, png_height, image_size, max_size)
        resized_widths.append(float(png_bw * scale))
        resized_heights.append(float(png_bh * scale))

    return {
        "train_rows": int(len(train_df)),
        "object_rows_excluding_class_14": int(len(object_df)),
        "valid_object_boxes": int(len(boxes)),
        "class_counts": {str(k): int(v) for k, v in sorted(class_counts.items())},
        "box_width_px": _numeric_summary(widths),
        "box_height_px": _numeric_summary(heights),
        "box_area_px": _numeric_summary(areas),
        "box_width_fraction_of_image": _numeric_summary(np.asarray(rel_widths, dtype=float)),
        "box_height_fraction_of_image": _numeric_summary(np.asarray(rel_heights, dtype=float)),
        "box_area_fraction_of_image": _numeric_summary(np.asarray(rel_areas, dtype=float)),
        "box_width_in_png_space_px": _numeric_summary(np.asarray(png_box_widths, dtype=float)),
        "box_height_in_png_space_px": _numeric_summary(np.asarray(png_box_heights, dtype=float)),
        "box_width_after_current_resize_px": _numeric_summary(np.asarray(resized_widths, dtype=float)),
        "box_height_after_current_resize_px": _numeric_summary(np.asarray(resized_heights, dtype=float)),
        "rows_missing_image_size": int(missing_size),
        "rows_missing_png_size": int(missing_png_size),
        "boxes_outside_declared_image": int(outside_image),
        "normalized_coordinate_like_boxes": int(normalized_like),
        "boxes_scaled_original_to_png": int(original_to_png_scaled),
    }


def _torchvision_resize_scale(width: float, height: float, image_size: int, max_size: int) -> float:
    min_original = min(width, height)
    max_original = max(width, height)
    scale = float(image_size) / max(min_original, 1e-9)
    if max_original * scale > max_size:
        scale = float(max_size) / max(max_original, 1e-9)
    return scale


def _recommend_resize(box_report: dict, image_size: int, max_size: int) -> dict:
    q = box_report.get("box_width_after_current_resize_px", {}).get("quantiles", {})
    p10_width = q.get("p10")
    p25_width = q.get("p25")
    p50_width = q.get("p50")
    outside = int(box_report.get("boxes_outside_declared_image", 0))
    normalized_like = int(box_report.get("normalized_coordinate_like_boxes", 0))

    notes = []
    suggested = {"image_size": image_size, "max_size": max_size}
    if normalized_like > 0:
        notes.append("Some boxes look normalized; inspect CSV columns before training.")
    if outside > 0:
        notes.append("Some boxes exceed declared image dimensions; inspect clipping or coordinate format.")
    if p10_width is not None and p10_width < 8:
        suggested = {"image_size": 1200, "max_size": 1800}
        notes.append("Small boxes become very small after resizing; try 1200/1800 if GPU memory allows.")
    elif p25_width is not None and p25_width < 12:
        suggested = {"image_size": 1024, "max_size": 1536}
        notes.append("Many boxes are small after resizing; try 1024/1536 as the next experiment.")
    elif p50_width is not None and p50_width > 80:
        notes.append("Current 800/1200 resize likely preserves enough box detail for V1.")
    else:
        notes.append("Current 800/1200 resize is a reasonable V1 baseline; tune after first validation result.")

    return {
        "current": {"image_size": image_size, "max_size": max_size},
        "suggested_next_experiment": suggested,
        "notes": notes,
    }


def _size_map_from_pil(*record_groups: Iterable[dict]) -> dict[str, tuple[float, float]]:
    size_map = {}
    for records in record_groups:
        for record in records:
            for variant in image_id_variants(record["image_id"]):
                size_map[variant] = (float(record["width"]), float(record["height"]))
    return size_map


def _count_missing_ids(image_ids: list[str], image_index: dict) -> int:
    return sum(resolve_image_path(image_id, image_index) is None for image_id in image_ids)


def _numeric_summary(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"count": 0}
    quantiles = np.percentile(values, [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100])
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "quantiles": {
            "min": float(quantiles[0]),
            "p01": float(quantiles[1]),
            "p05": float(quantiles[2]),
            "p10": float(quantiles[3]),
            "p25": float(quantiles[4]),
            "p50": float(quantiles[5]),
            "p75": float(quantiles[6]),
            "p90": float(quantiles[7]),
            "p95": float(quantiles[8]),
            "p99": float(quantiles[9]),
            "max": float(quantiles[10]),
        },
    }


def _common_sizes(sizes: Iterable[tuple[float, float]], limit: int = 10) -> list[dict]:
    counter = Counter((int(round(w)), int(round(h))) for w, h in sizes)
    return [{"width": w, "height": h, "count": count} for (w, h), count in counter.most_common(limit)]


def _without_private(report: dict) -> dict:
    return {key: value for key, value in report.items() if not key.startswith("_")}
