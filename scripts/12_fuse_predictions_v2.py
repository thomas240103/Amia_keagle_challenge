#!/usr/bin/env python
"""Fuse V2 scanner, global, and crop predictions."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.fuse_predictions import fuse_three_model_scores
from src.utils.config import apply_runtime_overrides, ensure_work_dir, load_config, output_path
from src.utils.logging import configure_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fuse V2 predictions.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--work-dir", default=None)
    parser.add_argument("--scanner-predictions", default=None)
    parser.add_argument("--global-scores", default=None)
    parser.add_argument("--crop-scores", default=None)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logger = configure_logger()
    config = apply_runtime_overrides(load_config(args.config), args.data_root, args.work_dir)
    ensure_work_dir(config)
    scanner_path = Path(args.scanner_predictions) if args.scanner_predictions else output_path(config, "lgcxr_fast_test_predictions.csv")
    global_path = Path(args.global_scores) if args.global_scores else output_path(config, "lgcxr_global_test_scores.csv")
    crop_path = Path(args.crop_scores) if args.crop_scores else output_path(config, "lgcxr_crop_test_scores.csv")
    output = Path(args.output) if args.output else Path(config.get("fusion", {}).get("output_path", output_path(config, "lgcxr_fused_test_predictions.csv")))

    scanner_df = pd.read_csv(scanner_path)
    global_df = pd.read_csv(global_path) if global_path.exists() else None
    crop_df = pd.read_csv(crop_path) if crop_path.exists() else None
    fusion_cfg = config.get("fusion", {})
    fused = fuse_three_model_scores(
        scanner_df,
        global_df,
        crop_df,
        scanner_weight=float(fusion_cfg.get("scanner_weight", 1.0)),
        global_weight=float(fusion_cfg.get("global_weight", 0.0)),
        crop_weight=float(fusion_cfg.get("crop_weight", 0.0)),
        default_threshold=float(fusion_cfg.get("default_threshold", config.get("inference", {}).get("conf_threshold", 0.03))),
        per_class_thresholds=fusion_cfg.get("per_class_thresholds", {}),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fused.to_csv(output, index=False)
    logger.info("Fused predictions saved to %s (%d rows)", output, len(fused))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
