#!/usr/bin/env python
"""Audit image dimensions and train box scale for resize decisions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dimension_audit import run_dimension_audit
from src.utils.config import apply_runtime_overrides, load_config
from src.utils.logging import configure_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit LG-CXR image dimensions and box scale.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--data-root", default=None, help="Override dataset root.")
    parser.add_argument("--work-dir", default=None, help="Override output directory.")
    parser.add_argument(
        "--max-images",
        type=int,
        default=0,
        help="Maximum train/test images to open for PIL dimension checks. Use 0 for all images.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logger = configure_logger()
    config = apply_runtime_overrides(load_config(args.config), data_root=args.data_root, work_dir=args.work_dir)
    report = run_dimension_audit(config, max_images=int(args.max_images))
    logger.info("Dimension audit saved to %s", report["output_path"])
    logger.info("Current resize: %s", json.dumps(report["recommendation"]["current"]))
    logger.info("Suggested next experiment: %s", json.dumps(report["recommendation"]["suggested_next_experiment"]))
    for note in report["recommendation"]["notes"]:
        logger.info("Recommendation note: %s", note)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
