#!/usr/bin/env python
"""Run dataset and submission preflight checks."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preflight import run_preflight
from src.utils.config import apply_runtime_overrides, load_config
from src.utils.logging import configure_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LG-CXR Faster R-CNN preflight checks.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--data-root", default=None, help="Override dataset root.")
    parser.add_argument("--work-dir", default=None, help="Override output directory.")
    parser.add_argument("--smoke-test", action="store_true", help="Apply smoke-test config overrides.")
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
    result = run_preflight(config)
    logger.info("Preflight passed with %d checks.", len(result["checks"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
