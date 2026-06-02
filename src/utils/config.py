"""Config loading and runtime override helpers."""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a YAML mapping: {path}")
    return config


def apply_runtime_overrides(
    config: dict[str, Any],
    data_root: str | None = None,
    work_dir: str | None = None,
    smoke_test: bool = False,
) -> dict[str, Any]:
    cfg = copy.deepcopy(config)

    env_data_root = os.environ.get("LGCXR_DATA_ROOT")
    env_work_dir = os.environ.get("LGCXR_WORK_DIR")

    if env_data_root:
        cfg["data_root"] = env_data_root
    if env_work_dir:
        cfg["work_dir"] = env_work_dir
    if data_root:
        cfg["data_root"] = data_root
    if work_dir:
        cfg["work_dir"] = work_dir
    if smoke_test:
        cfg["smoke_test"] = True

    cfg = normalize_work_dir_paths(cfg)
    if cfg.get("smoke_test"):
        cfg = apply_smoke_defaults(cfg)
    return cfg


def normalize_work_dir_paths(config: dict[str, Any]) -> dict[str, Any]:
    cfg = copy.deepcopy(config)
    work_dir = Path(str(cfg["work_dir"]))

    scanner = cfg.setdefault("scanner", {})
    scanner["best_ckpt"] = str(work_dir / Path(scanner.get("best_ckpt", "lgcxr_scanner_fasterrcnn_best.pth")).name)
    scanner["last_ckpt"] = str(work_dir / Path(scanner.get("last_ckpt", "lgcxr_scanner_fasterrcnn_last.pth")).name)

    global_classifier = cfg.setdefault("global_classifier", {})
    global_classifier["checkpoint"] = str(
        work_dir / Path(global_classifier.get("checkpoint", "lgcxr_global_classifier_best.pth")).name
    )

    crop_classifier = cfg.setdefault("crop_classifier", {})
    crop_classifier["checkpoint"] = str(
        work_dir / Path(crop_classifier.get("checkpoint", "lgcxr_crop_classifier_best.pth")).name
    )

    fusion = cfg.setdefault("fusion", {})
    if "output_path" in fusion:
        fusion["output_path"] = str(work_dir / Path(fusion.get("output_path", "lgcxr_fused_test_predictions.csv")).name)

    submission = cfg.setdefault("submission", {})
    submission["output_path"] = str(work_dir / Path(submission.get("output_path", "submission.csv")).name)
    return cfg


def apply_smoke_defaults(config: dict[str, Any]) -> dict[str, Any]:
    cfg = copy.deepcopy(config)
    scanner = cfg.setdefault("scanner", {})
    scanner["epochs"] = min(int(scanner.get("epochs", 1)), 1)
    scanner["num_workers"] = 0
    scanner["batch_size"] = min(int(scanner.get("batch_size", 1)), 2)
    return cfg


def ensure_work_dir(config: dict[str, Any]) -> Path:
    work_dir = Path(str(config["work_dir"]))
    work_dir.mkdir(parents=True, exist_ok=True)
    return work_dir


def output_path(config: dict[str, Any], filename: str) -> Path:
    return Path(str(config["work_dir"])) / filename
