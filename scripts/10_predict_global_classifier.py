#!/usr/bin/env python
"""Predict optional V2 global classifier scores."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.classification_dataset import GlobalCXRDataset
from src.data.columns import infer_detection_columns, infer_image_id_column
from src.data.discovery import discover_dataset, load_csv
from src.data.image_paths import build_image_index
from src.data.splits import ordered_unique, sample_submission_image_ids
from src.models.resnet18_classifier import build_resnet18_classifier
from src.utils.accelerator import log_accelerator, maybe_wrap_data_parallel
from src.utils.checkpoints import load_checkpoint
from src.utils.config import apply_runtime_overrides, ensure_work_dir, load_config, output_path
from src.utils.logging import configure_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict global classifier scores.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--work-dir", default=None)
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logger = configure_logger()
    config = apply_runtime_overrides(load_config(args.config), args.data_root, args.work_dir, args.smoke_test)
    ensure_work_dir(config)
    cfg = config["global_classifier"]
    if not bool(cfg.get("enabled", False)):
        logger.info("Global classifier disabled; skipping prediction.")
        return 0
    ckpt_path = Path(cfg["checkpoint"])
    if not ckpt_path.exists():
        logger.warning("Global classifier checkpoint missing; skipping: %s", ckpt_path)
        return 0

    discovery = discover_dataset(config["data_root"])
    train_df = load_csv(discovery.train_csv)
    sample_df = load_csv(discovery.sample_submission)
    columns = infer_detection_columns(train_df)
    test_ids = _test_ids(discovery, sample_df, config)
    image_index = build_image_index(discovery.test_image_paths or discovery.image_paths)

    dataset = GlobalCXRDataset(train_df, test_ids, image_index, columns, int(cfg["image_size"]))
    loader = DataLoader(dataset, batch_size=int(cfg.get("batch_size", 16)), shuffle=False, num_workers=int(cfg.get("num_workers", 2)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_accelerator(logger, "Global classifier prediction")
    model = build_resnet18_classifier(num_classes=14, pretrained=False).to(device)
    model = maybe_wrap_data_parallel(model, cfg, logger, "Global classifier prediction")
    load_checkpoint(ckpt_path, model, map_location=device)
    model.eval()

    records = []
    with torch.no_grad():
        for images, _, image_ids in loader:
            probs = torch.sigmoid(model(images.to(device))).cpu()
            for image_id, row in zip(image_ids, probs):
                for class_id, score in enumerate(row.tolist()):
                    records.append({"image_id": str(image_id), "class_id": class_id, "global_score": float(score)})

    out = output_path(config, "lgcxr_global_test_scores.csv")
    pd.DataFrame(records).to_csv(out, index=False)
    logger.info("Global scores saved to %s", out)
    return 0


def _test_ids(discovery, sample_df, config) -> list[str]:
    if discovery.test_csv is not None:
        try:
            test_df = load_csv(discovery.test_csv)
            col = infer_image_id_column(test_df)
            ids = ordered_unique([str(v) for v in test_df[col].dropna().astype(str).tolist()])
            if ids:
                return ids[: int(config.get("smoke_max_test_images", 100))] if config.get("smoke_test") else ids
        except Exception:
            pass
    return sample_submission_image_ids(sample_df, bool(config.get("smoke_test", False)), int(config.get("smoke_max_test_images", 100)))


if __name__ == "__main__":
    raise SystemExit(main())
