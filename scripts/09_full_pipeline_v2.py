#!/usr/bin/env python
"""Run the V2 three-model pipeline."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import apply_runtime_overrides, load_config, output_path
from src.utils.logging import configure_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run V2 scanner + global + crop pipeline.")
    parser.add_argument("--config", default="configs/v2_three_model.yaml")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--work-dir", default=None)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--force-train", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logger = configure_logger()
    config = apply_runtime_overrides(load_config(args.config), args.data_root, args.work_dir, args.smoke_test)
    common = ["--config", args.config]
    if args.data_root:
        common += ["--data-root", args.data_root]
    if args.work_dir:
        common += ["--work-dir", args.work_dir]
    if args.smoke_test:
        common += ["--smoke-test"]

    _run("preflight", [sys.executable, "scripts/00_preflight.py", *common])
    _run("dimension audit", [sys.executable, "scripts/05_audit_dimensions.py", *common])

    scanner_ckpt = Path(config["scanner"]["best_ckpt"])
    if args.force_train or not scanner_ckpt.exists():
        _run("train scanner", [sys.executable, "scripts/01_train_scanner.py", *common, "--skip-preflight"])
    else:
        logger.info("Scanner checkpoint exists; skipping training: %s", scanner_ckpt)
    _run("predict scanner", [sys.executable, "scripts/02_predict_scanner.py", *common])

    global_ckpt = Path(config["global_classifier"]["checkpoint"])
    if bool(config["global_classifier"].get("enabled", False)) and (args.force_train or not global_ckpt.exists()):
        _run("train global classifier", [sys.executable, "scripts/07_train_global_classifier.py", *common])
    _run("predict global classifier", [sys.executable, "scripts/10_predict_global_classifier.py", *common])

    crop_ckpt = Path(config["crop_classifier"]["checkpoint"])
    if bool(config["crop_classifier"].get("enabled", False)) and (args.force_train or not crop_ckpt.exists()):
        _run("train crop classifier", [sys.executable, "scripts/08_train_crop_classifier.py", *common])
    _run("predict crop classifier", [sys.executable, "scripts/11_predict_crop_classifier.py", *common])

    _run("fuse predictions", [sys.executable, "scripts/12_fuse_predictions_v2.py", *common])
    fused_path = str(Path(config.get("fusion", {}).get("output_path", output_path(config, "lgcxr_fused_test_predictions.csv"))))
    make_submission_common = [item for item in common if item != "--smoke-test"]
    _run("make fused submission", [sys.executable, "scripts/03_make_submission.py", *make_submission_common, "--prediction-csv", fused_path])
    logger.info("V2 pipeline complete.")
    return 0


def _run(name: str, command: list[str]) -> None:
    print(f"\n=== {name} ===")
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


if __name__ == "__main__":
    raise SystemExit(main())
