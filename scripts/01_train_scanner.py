#!/usr/bin/env python
"""Train the Faster R-CNN scanner."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.columns import infer_detection_columns
from src.data.dataset import CXRDetectionDataset
from src.data.discovery import discover_dataset, load_csv
from src.data.image_paths import build_image_index, image_ids_from_paths
from src.data.image_sizes import load_image_size_map
from src.data.preflight import run_preflight
from src.data.splits import make_train_val_split
from src.models.fasterrcnn import build_fasterrcnn
from src.train.train_scanner import detection_collate, evaluate_detector_map, make_optimizer, train_one_epoch
from src.utils.checkpoints import load_checkpoint, save_checkpoint
from src.utils.accelerator import log_accelerator
from src.utils.config import apply_runtime_overrides, ensure_work_dir, load_config, output_path
from src.utils.logging import configure_logger
from src.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LG-CXR Faster R-CNN scanner.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--data-root", default=None, help="Override dataset root.")
    parser.add_argument("--work-dir", default=None, help="Override output directory.")
    parser.add_argument("--smoke-test", action="store_true", help="Train on small subsets for one epoch.")
    parser.add_argument("--skip-preflight", action="store_true", help="Skip preflight because it already passed.")
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
    work_dir = ensure_work_dir(config)
    set_seed(int(config.get("seed", 42)))

    if not args.skip_preflight:
        logger.info("Running preflight before training.")
        run_preflight(config)

    discovery = discover_dataset(config["data_root"])
    if discovery.train_csv is None:
        raise RuntimeError("No train CSV found after preflight.")
    train_df = load_csv(discovery.train_csv)
    columns = infer_detection_columns(train_df)
    original_size_map, size_columns = load_image_size_map(discovery.img_size_csv)
    if original_size_map:
        logger.info("Loaded original image sizes from %s using columns %s", discovery.img_size_csv, size_columns)
    else:
        logger.warning("No img_size.csv mapping loaded; training boxes will be used without original-to-PNG scaling.")
    train_image_paths = discovery.train_image_paths or discovery.image_paths
    image_index = build_image_index(train_image_paths)
    all_train_image_ids = image_ids_from_paths(train_image_paths) if discovery.train_image_paths else None

    train_ids, val_ids = make_train_val_split(
        train_df,
        columns,
        val_size=float(config.get("val_size", 0.2)),
        seed=int(config.get("seed", 42)),
        all_image_ids=all_train_image_ids,
        smoke_test=bool(config.get("smoke_test", False)),
        smoke_max_train_images=int(config.get("smoke_max_train_images", 500)),
        smoke_max_val_images=int(config.get("smoke_max_val_images", 100)),
    )
    logger.info("Train images: %d | Val images: %d", len(train_ids), len(val_ids))

    no_finding_class_id = int(config["submission"]["no_finding_class_id"])
    train_dataset = CXRDetectionDataset(
        train_df,
        train_ids,
        image_index,
        columns,
        no_finding_class_id,
        original_size_map=original_size_map,
    )
    val_dataset = CXRDetectionDataset(
        train_df,
        val_ids,
        image_index,
        columns,
        no_finding_class_id,
        original_size_map=original_size_map,
    )

    scanner_cfg = config["scanner"]
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(scanner_cfg["batch_size"]),
        shuffle=True,
        num_workers=int(scanner_cfg["num_workers"]),
        collate_fn=detection_collate,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=int(scanner_cfg["num_workers"]),
        collate_fn=detection_collate,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_accelerator(logger, "Scanner")
    logger.info("Using device: %s", device)
    model = build_fasterrcnn(
        num_classes=15,
        image_size=int(scanner_cfg["image_size"]),
        max_size=int(scanner_cfg.get("max_size", round(int(scanner_cfg["image_size"]) * 1.5))),
    )
    model.to(device)

    optimizer = make_optimizer(model, lr=float(scanner_cfg["lr"]), weight_decay=float(scanner_cfg["weight_decay"]))
    amp_enabled = bool(scanner_cfg.get("amp", True) and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    start_epoch = 0
    best_score = -1.0
    last_ckpt = Path(scanner_cfg["last_ckpt"])
    if bool(scanner_cfg.get("resume", True)) and last_ckpt.exists():
        logger.info("Resuming from last checkpoint: %s", last_ckpt)
        checkpoint = load_checkpoint(last_ckpt, model, optimizer=optimizer, scaler=scaler, map_location=device)
        start_epoch = int(checkpoint.get("epoch", -1)) + 1
        best_score = float(checkpoint.get("best_score", -1.0))

    history = []
    epochs = int(scanner_cfg["epochs"])
    for epoch in range(start_epoch, epochs):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            amp=amp_enabled,
            grad_clip=scanner_cfg.get("grad_clip"),
            scaler=scaler,
            epoch=epoch,
        )
        if len(val_dataset) > 0:
            metrics = evaluate_detector_map(
                model,
                val_loader,
                val_dataset,
                device,
                iou_threshold=float(config["evaluation"]["voc_iou_threshold"]),
                conf_threshold=float(config["inference"]["conf_threshold"]),
            )
            score = float(metrics["map"])
        else:
            metrics = {"map": 0.0, "per_class_ap": {}, "iou_threshold": config["evaluation"]["voc_iou_threshold"]}
            score = -train_loss

        save_checkpoint(scanner_cfg["last_ckpt"], model, optimizer, scaler, epoch, best_score, config)
        if score > best_score:
            best_score = score
            save_checkpoint(scanner_cfg["best_ckpt"], model, optimizer, scaler, epoch, best_score, config)

        row = {"epoch": epoch, "train_loss": train_loss, "val_map_proxy": score, "metrics": metrics}
        history.append(row)
        logger.info("Epoch %d | loss %.5f | val proxy %.5f | best %.5f", epoch, train_loss, score, best_score)

    if not Path(scanner_cfg["best_ckpt"]).exists():
        logger.info("Best checkpoint was not present; saving current model as best.")
        save_checkpoint(scanner_cfg["best_ckpt"], model, optimizer, scaler, max(start_epoch - 1, 0), best_score, config)

    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "train_csv": str(discovery.train_csv),
        "train_images": len(train_ids),
        "val_images": len(val_ids),
        "device": str(device),
        "best_score": best_score,
        "history": history,
        "config": config,
    }
    summary_path = output_path(config, "lgcxr_scanner_training_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Training summary saved to %s", summary_path)
    logger.info("Best checkpoint: %s", scanner_cfg["best_ckpt"])
    logger.info("Last checkpoint: %s", scanner_cfg["last_ckpt"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
