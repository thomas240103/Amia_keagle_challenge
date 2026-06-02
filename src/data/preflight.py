"""Preflight checks that must pass before training."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from PIL import Image

from src.utils.config import ensure_work_dir
from src.utils.visualization import draw_debug_annotation

from .columns import (
    extract_boxes_and_labels,
    infer_detection_columns,
    infer_image_id_column,
    infer_image_size_columns,
    object_rows,
    validate_annotation_rows,
)
from .discovery import discover_dataset, load_csv
from .image_paths import build_image_index, image_id_variants, image_ids_from_paths, resolve_image_path
from .image_sizes import build_image_size_map, lookup_original_size, scale_original_boxes_to_png
from .splits import sample_submission_image_ids


def run_preflight(config: dict) -> dict:
    work_dir = ensure_work_dir(config)
    root = Path(str(config["data_root"]))
    no_finding_class_id = int(config["submission"]["no_finding_class_id"])
    checks: list[dict] = []

    def add(name: str, ok: bool, detail: str, critical: bool = True) -> None:
        checks.append({"name": name, "ok": bool(ok), "detail": detail, "critical": bool(critical)})

    add("dataset_exists", root.exists(), str(root))
    if not root.exists():
        return _finish(work_dir, checks)

    discovery = discover_dataset(root)
    add("train_folder_found", discovery.train_dir is not None, str(discovery.train_dir))
    add("test_folder_found", discovery.test_dir is not None, str(discovery.test_dir))
    add("explicit_train_csv_found", discovery.train_csv is not None and discovery.train_csv.name == "train.csv", str(discovery.train_csv))
    add("explicit_test_csv_found", discovery.test_csv is not None, str(discovery.test_csv))
    add("explicit_img_size_csv_found", discovery.img_size_csv is not None, str(discovery.img_size_csv), critical=False)
    add(
        "explicit_sample_submission_found",
        discovery.sample_submission is not None and discovery.sample_submission.name == "sample_submission.csv",
        str(discovery.sample_submission),
    )
    add("csv_files_found", len(discovery.csv_paths) > 0, f"{len(discovery.csv_paths)} CSV files")
    loaded_csvs = 0
    for csv_path in discovery.csv_paths:
        try:
            pd.read_csv(csv_path, nrows=3)
            loaded_csvs += 1
        except Exception as exc:
            add("csv_load_failed", False, f"{csv_path}: {exc}")
    add("csv_files_load", loaded_csvs == len(discovery.csv_paths), f"{loaded_csvs}/{len(discovery.csv_paths)} loaded")

    add("train_csv_identified", discovery.train_csv is not None, str(discovery.train_csv))
    add("sample_submission_identified", discovery.sample_submission is not None, str(discovery.sample_submission))
    add("train_images_found", len(discovery.train_image_paths) > 0, f"{len(discovery.train_image_paths)} train images")
    add("test_images_found", len(discovery.test_image_paths) > 0, f"{len(discovery.test_image_paths)} test images")
    add("images_found", len(discovery.image_paths) > 0, f"{len(discovery.image_paths)} total image files")
    if discovery.train_csv is None or discovery.sample_submission is None:
        return _finish(work_dir, checks)

    train_df = load_csv(discovery.train_csv)
    sample_df = load_csv(discovery.sample_submission)
    test_df = load_csv(discovery.test_csv) if discovery.test_csv is not None else None
    img_size_df = load_csv(discovery.img_size_csv) if discovery.img_size_csv is not None else None

    try:
        columns = infer_detection_columns(train_df)
        add("required_columns_inferred", True, json.dumps(columns.to_dict()))
    except Exception as exc:
        add("required_columns_inferred", False, str(exc))
        return _finish(work_dir, checks)

    add(
        "sample_submission_columns",
        list(sample_df.columns) == ["image_id", "PredictionString"],
        f"{list(sample_df.columns)}",
    )
    if test_df is not None:
        try:
            test_image_col = infer_image_id_column(test_df)
            add("test_csv_image_column_inferred", True, test_image_col)
        except Exception as exc:
            add("test_csv_image_column_inferred", False, str(exc))
            test_image_col = None
    else:
        test_image_col = None

    if img_size_df is not None:
        try:
            size_columns = infer_image_size_columns(img_size_df)
            size_map = build_image_size_map(img_size_df, size_columns)
            add("img_size_columns_inferred", True, json.dumps(size_columns.to_dict()))
            add("img_size_rows_loaded", len(size_map) > 0, f"{len(size_map)} image-size keys", critical=False)
        except Exception as exc:
            size_columns = None
            size_map = {}
            add("img_size_columns_inferred", False, str(exc), critical=False)
    else:
        size_columns = None
        size_map = {}

    validation = validate_annotation_rows(train_df, columns, no_finding_class_id)
    add("bounding_boxes_valid", validation["invalid_object_rows"] == 0, json.dumps(validation))
    if size_map:
        add("boxes_within_img_size", *_validate_boxes_against_size(train_df, columns, size_map, no_finding_class_id), critical=False)
    add(
        "class_14_no_finding_only",
        True,
        f"{validation['no_finding_rows']} rows with class {no_finding_class_id} will be ignored for detector training",
    )

    train_image_paths = discovery.train_image_paths or discovery.image_paths
    test_image_paths = discovery.test_image_paths or discovery.image_paths
    train_image_index = build_image_index(train_image_paths)
    test_image_index = build_image_index(test_image_paths)

    train_ids_from_csv = [str(v) for v in train_df[columns.image_id].dropna().astype(str).unique().tolist()]
    train_ids = image_ids_from_paths(train_image_paths) if discovery.train_image_paths else train_ids_from_csv
    test_ids = sample_submission_image_ids(sample_df)
    test_ids_from_csv = (
        [str(v) for v in test_df[test_image_col].dropna().astype(str).unique().tolist()]
        if test_df is not None and test_image_col is not None
        else []
    )

    train_resolved = [resolve_image_path(image_id, train_image_index) for image_id in train_ids]
    train_csv_resolved = [resolve_image_path(image_id, train_image_index) for image_id in train_ids_from_csv]
    test_resolved = [resolve_image_path(image_id, test_image_index) for image_id in test_ids]
    add("train_folder_image_ids_used", len(train_ids) > 0, f"{len(train_ids)} train IDs from train/ folder")
    add("train_images_resolve", all(p is not None for p in train_resolved), f"{sum(p is not None for p in train_resolved)}/{len(train_ids)}")
    add("train_csv_rows_resolve_to_train_folder", all(p is not None for p in train_csv_resolved), f"{sum(p is not None for p in train_csv_resolved)}/{len(train_ids_from_csv)}")
    add("test_images_resolve", all(p is not None for p in test_resolved), f"{sum(p is not None for p in test_resolved)}/{len(test_ids)}")
    if test_ids_from_csv:
        test_csv_resolved = [resolve_image_path(image_id, test_image_index) for image_id in test_ids_from_csv]
        add("test_csv_rows_resolve_to_test_folder", all(p is not None for p in test_csv_resolved), f"{sum(p is not None for p in test_csv_resolved)}/{len(test_ids_from_csv)}")
        add("test_csv_matches_sample_ids", _same_id_set(test_ids_from_csv, test_ids), f"test.csv={len(test_ids_from_csv)} sample_submission={len(test_ids)}", critical=False)

    add("train_images_open", *_open_sample(train_resolved, 50, "train"))
    add("test_images_open", *_open_sample(test_resolved, 20, "test"))
    if size_map:
        add(
            "coordinate_space_scaling_required",
            True,
            _coordinate_space_summary(train_ids[:50], train_image_index, size_map),
            critical=False,
        )

    debug_saved = False
    debug_detail = "No resolvable training image"
    for image_id in train_ids:
        path = resolve_image_path(image_id, train_image_index)
        if path is None:
            continue
        rows = train_df[train_df[columns.image_id].astype(str) == str(image_id)]
        rows = object_rows(rows, columns, no_finding_class_id)
        boxes, labels = extract_boxes_and_labels(rows, columns)
        with Image.open(path) as image:
            width, height = image.size
        boxes, _ = scale_original_boxes_to_png(boxes, image_id, size_map, width, height)
        debug_path = work_dir / "debug_preflight_annotation.png"
        draw_debug_annotation(path, boxes[:10], labels[:10].astype(int).tolist(), debug_path)
        debug_saved = True
        debug_detail = str(debug_path)
        break
    add("debug_image_saved", debug_saved, debug_detail)

    return _finish(work_dir, checks)


def _open_sample(paths, max_count: int, label: str) -> tuple[bool, str]:
    sample = [p for p in paths if p is not None][:max_count]
    if not sample:
        return False, f"No {label} images resolved"
    opened = 0
    failures = []
    for path in sample:
        try:
            with Image.open(path) as image:
                image.verify()
            opened += 1
        except Exception as exc:
            failures.append(f"{path}: {exc}")
    ok = opened == len(sample)
    detail = f"{opened}/{len(sample)} opened"
    if failures:
        detail += f"; first failure: {failures[0]}"
    return ok, detail


def _validate_boxes_against_size(train_df, columns, size_map, no_finding_class_id: int) -> tuple[bool, str]:
    rows = object_rows(train_df, columns, no_finding_class_id)
    if len(rows) == 0:
        return True, "No object rows to validate against img_size.csv"
    boxes, _ = extract_boxes_and_labels(rows, columns)
    missing_size = 0
    outside = 0
    normalized_like = 0
    checked = 0
    for (_, row), box in zip(rows.iterrows(), boxes):
        image_id = str(row[columns.image_id])
        size = lookup_original_size(image_id, size_map)
        if size is None:
            missing_size += 1
            continue
        width, height = size
        x1, y1, x2, y2 = [float(v) for v in box]
        checked += 1
        if max(x1, y1, x2, y2) <= 1.5:
            normalized_like += 1
        if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
            outside += 1
    ok = outside == 0
    detail = (
        f"checked={checked}, outside_image={outside}, missing_size={missing_size}, "
        f"normalized_like={normalized_like}"
    )
    return ok, detail


def _coordinate_space_summary(image_ids, image_index, size_map) -> str:
    checked = 0
    different = 0
    examples = []
    for image_id in image_ids:
        path = resolve_image_path(image_id, image_index)
        original_size = lookup_original_size(image_id, size_map)
        if path is None or original_size is None:
            continue
        with Image.open(path) as image:
            png_width, png_height = image.size
        orig_width, orig_height = original_size
        checked += 1
        if abs(orig_width - png_width) > 1 or abs(orig_height - png_height) > 1:
            different += 1
            if len(examples) < 3:
                examples.append(f"{image_id}: original={int(orig_width)}x{int(orig_height)} png={png_width}x{png_height}")
    return f"{different}/{checked} checked train images need original-to-PNG box scaling; examples: {examples}"


def _same_id_set(left: list[str], right: list[str]) -> bool:
    left_keys = {_canonical_id(v) for v in left}
    right_keys = {_canonical_id(v) for v in right}
    return left_keys == right_keys


def _canonical_id(value: str) -> str:
    normalized = str(value).strip().strip('"').strip("'").replace("\\", "/")
    return Path(normalized).stem.lower()


def _finish(work_dir: Path, checks: list[dict]) -> dict:
    ok = all(check["ok"] for check in checks if check["critical"])
    result = {"ok": ok, "checks": checks}
    status_path = work_dir / "lgcxr_preflight_status.json"
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    if not ok:
        failed = [check for check in checks if check["critical"] and not check["ok"]]
        details = "; ".join(f"{check['name']}: {check['detail']}" for check in failed)
        raise RuntimeError(f"Preflight failed: {details}")
    return result
