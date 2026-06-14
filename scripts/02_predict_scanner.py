#!/usr/bin/env python
"""Predict validation and test detections with the Faster R-CNN scanner."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.columns import infer_detection_columns, infer_image_id_column
from src.data.dataset import CXRImageDataset
from src.data.discovery import discover_dataset, load_csv
from src.data.image_paths import build_image_index, image_ids_from_paths
from src.data.image_sizes import load_image_size_map
from src.data.splits import make_train_val_split, ordered_unique, sample_submission_image_ids
from src.inference.predict_scanner import image_collate, predict_dataset, save_predictions
from src.models.fasterrcnn import build_fasterrcnn
from src.utils.checkpoints import load_checkpoint
from src.utils.config import apply_runtime_overrides, ensure_work_dir, load_config, output_path
from src.utils.logging import configure_logger
from src.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict LG-CXR Faster R-CNN detections.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--data-root", default=None, help="Override dataset root.")
    parser.add_argument("--work-dir", default=None, help="Override output directory.")
    parser.add_argument("--smoke-test", action="store_true", help="Predict on small subsets.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logger = configure_logger()
    config = apply_runtime_overrides(
        load_config(args.config),
        data_root=args.data_root,
        work_dir=args.work_dir,
        smoke_test=args.smoke_test,
    )
    ensure_work_dir(config)
    set_seed(int(config.get("seed", 42)))

    scanner_cfg = config["scanner"]
    ckpt_path = Path(scanner_cfg["best_ckpt"])
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Best scanner checkpoint not found: {ckpt_path}")

    discovery = discover_dataset(config["data_root"])
    if discovery.train_csv is None or discovery.sample_submission is None:
        raise RuntimeError("Train CSV and sample submission are required for scanner prediction.")

    train_df = load_csv(discovery.train_csv)
    sample_df = load_csv(discovery.sample_submission)
    columns = infer_detection_columns(train_df)
    original_size_map, size_columns = load_image_size_map(discovery.img_size_csv)
    if original_size_map:
        logger.info("Loaded original image sizes from %s using columns %s", discovery.img_size_csv, size_columns)
    else:
        logger.warning("No img_size.csv mapping loaded; predictions will stay in PNG/model image space.")
    train_image_paths = discovery.train_image_paths or discovery.image_paths
    test_image_paths = discovery.test_image_paths or discovery.image_paths
    train_image_index = build_image_index(train_image_paths)
    test_image_index = build_image_index(test_image_paths)
    all_train_image_ids = image_ids_from_paths(train_image_paths) if discovery.train_image_paths else None

    _, val_ids = make_train_val_split(
        train_df,
        columns,
        val_size=float(config.get("val_size", 0.2)),
        seed=int(config.get("seed", 42)),
        all_image_ids=all_train_image_ids,
        smoke_test=bool(config.get("smoke_test", False)),
        smoke_max_train_images=int(config.get("smoke_max_train_images", 500)),
        smoke_max_val_images=int(config.get("smoke_max_val_images", 100)),
    )
    test_ids = _test_ids_from_metadata_or_sample(discovery, sample_df, config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_fasterrcnn(
        num_classes=15,
        image_size=int(scanner_cfg["image_size"]),
        max_size=int(scanner_cfg.get("max_size", round(int(scanner_cfg["image_size"]) * 1.5))),
    )
    load_checkpoint(ckpt_path, model, map_location=device)
    model.to(device)

    inference_cfg = config["inference"]
    val_df = _predict_ids(
        model,
        val_ids,
        train_image_index,
        device,
        inference_cfg,
        int(scanner_cfg["num_workers"]),
        original_size_map=original_size_map,
    )
    test_df = _predict_ids(
        model,
        test_ids,
        test_image_index,
        device,
        inference_cfg,
        int(scanner_cfg["num_workers"]),
        original_size_map=original_size_map,
    )

    val_path = save_predictions(val_df, output_path(config, "lgcxr_fast_val_predictions.csv"))
    test_path = save_predictions(test_df, output_path(config, "lgcxr_fast_test_predictions.csv"))
    logger.info("Validation predictions saved to %s (%d rows)", val_path, len(val_df))
    logger.info("Test predictions saved to %s (%d rows)", test_path, len(test_df))
    return 0


def _predict_ids(model, image_ids, image_index, device, inference_cfg, num_workers: int, original_size_map=None) -> pd.DataFrame:
    if not image_ids:
        return pd.DataFrame(columns=["image_id", "class_id", "confidence", "xmin", "ymin", "xmax", "ymax"])
    dataset = CXRImageDataset(image_ids, image_index)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=num_workers, collate_fn=image_collate)
    return predict_dataset(
        model,
        loader,
        device,
        conf_threshold=float(inference_cfg["conf_threshold"]),
        nms_iou=float(inference_cfg["nms_iou"]),
        max_det_per_image=int(inference_cfg["max_det_per_image"]),
        original_size_map=original_size_map,
    )


def _test_ids_from_metadata_or_sample(discovery, sample_df, config) -> list[str]:
    smoke_test = bool(config.get("smoke_test", False))
    smoke_max_test_images = int(config.get("smoke_max_test_images", 100))
    if discovery.test_csv is not None:
        test_df = load_csv(discovery.test_csv)
        try:
            image_col = infer_image_id_column(test_df)
            test_ids = ordered_unique([str(v) for v in test_df[image_col].dropna().astype(str).tolist()])
            if smoke_test:
                test_ids = test_ids[:smoke_max_test_images]
            if test_ids:
                return test_ids
        except Exception:
            pass
    return sample_submission_image_ids(sample_df, smoke_test=smoke_test, smoke_max_test_images=smoke_max_test_images)


if __name__ == "__main__":
    raise SystemExit(main())
