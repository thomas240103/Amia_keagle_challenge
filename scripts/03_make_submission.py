#!/usr/bin/env python
"""Create Kaggle submission.csv from test predictions."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.discovery import discover_dataset, load_csv
from src.inference.make_submission import make_submission
from src.utils.config import apply_runtime_overrides, ensure_work_dir, load_config, output_path
from src.utils.logging import configure_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create LG-CXR submission.csv.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--data-root", default=None, help="Override dataset root.")
    parser.add_argument("--work-dir", default=None, help="Override output directory.")
    parser.add_argument("--prediction-csv", default=None, help="Override prediction CSV. Defaults to lgcxr_fast_test_predictions.csv.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logger = configure_logger()
    config = apply_runtime_overrides(load_config(args.config), data_root=args.data_root, work_dir=args.work_dir)
    ensure_work_dir(config)

    discovery = discover_dataset(config["data_root"])
    if discovery.sample_submission is None:
        raise RuntimeError("Could not identify sample submission CSV.")

    sample_df = load_csv(discovery.sample_submission)
    prediction_path = Path(args.prediction_csv) if args.prediction_csv else output_path(config, "lgcxr_fast_test_predictions.csv")
    if not prediction_path.exists():
        raise FileNotFoundError(f"Test prediction CSV not found: {prediction_path}")
    predictions = pd.read_csv(prediction_path)

    output = config["submission"]["output_path"]
    submission = make_submission(
        sample_df,
        predictions,
        output,
        no_finding_string=str(config["submission"]["no_finding_string"]),
    )
    logger.info("Submission saved to %s (%d rows)", output, len(submission))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
