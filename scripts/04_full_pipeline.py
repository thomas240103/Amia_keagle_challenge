#!/usr/bin/env python
"""Run preflight, training, prediction, and submission creation."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import apply_runtime_overrides, load_config
from src.utils.logging import configure_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full LG-CXR Faster R-CNN pipeline.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--data-root", default=None, help="Override dataset root.")
    parser.add_argument("--work-dir", default=None, help="Override output directory.")
    parser.add_argument("--smoke-test", action="store_true", help="Run small one-epoch smoke pipeline.")
    parser.add_argument("--force-train", action="store_true", help="Train even if best checkpoint already exists.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logger = configure_logger()
    config_path = str(Path(args.config))
    config = apply_runtime_overrides(
        load_config(config_path),
        data_root=args.data_root,
        work_dir=args.work_dir,
        smoke_test=args.smoke_test,
    )

    common = ["--config", config_path]
    if args.data_root:
        common += ["--data-root", args.data_root]
    if args.work_dir:
        common += ["--work-dir", args.work_dir]
    if args.smoke_test:
        common += ["--smoke-test"]

    _run("preflight", [sys.executable, "scripts/00_preflight.py", *common])

    best_ckpt = Path(config["scanner"]["best_ckpt"])
    if args.force_train or not best_ckpt.exists():
        logger.info("Training scanner because best checkpoint is missing or force-train was requested.")
        _run("train scanner", [sys.executable, "scripts/01_train_scanner.py", *common, "--skip-preflight"])
    else:
        logger.info("Best checkpoint exists; skipping training: %s", best_ckpt)

    _run("predict scanner", [sys.executable, "scripts/02_predict_scanner.py", *common])
    make_submission_common = [item for item in common if item != "--smoke-test"]
    _run("make submission", [sys.executable, "scripts/03_make_submission.py", *make_submission_common])
    logger.info("Full pipeline complete.")
    return 0


def _run(name: str, command: list[str]) -> None:
    print(f"\n=== {name} ===")
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


if __name__ == "__main__":
    raise SystemExit(main())
