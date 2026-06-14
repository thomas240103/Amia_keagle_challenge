#!/usr/bin/env python
"""Train optional V2 crop ResNet18 classifier."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.classification_dataset import CropCXRDataset
from src.data.columns import infer_detection_columns
from src.data.discovery import discover_dataset, load_csv
from src.data.image_paths import build_image_index, image_ids_from_paths
from src.data.image_sizes import load_image_size_map
from src.data.splits import make_train_val_split
from src.models.resnet18_classifier import build_resnet18_classifier
from src.train.train_classifier import evaluate_multiclass_loss, train_multiclass_epoch
from src.utils.accelerator import log_accelerator, maybe_wrap_data_parallel
from src.utils.checkpoints import load_checkpoint, save_checkpoint
from src.utils.config import apply_runtime_overrides, ensure_work_dir, load_config, output_path
from src.utils.logging import configure_logger
from src.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train crop ResNet18 classifier.")
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
    set_seed(int(config.get("seed", 42)))
    cfg = config["crop_classifier"]
    if not bool(cfg.get("enabled", False)):
        logger.info("Crop classifier disabled; skipping.")
        return 0

    discovery = discover_dataset(config["data_root"])
    if discovery.train_csv is None:
        raise RuntimeError("train.csv is required.")
    train_df = load_csv(discovery.train_csv)
    columns = infer_detection_columns(train_df)
    original_size_map, _ = load_image_size_map(discovery.img_size_csv)
    if not original_size_map:
        raise RuntimeError("img_size.csv is required for crop classifier coordinate scaling.")

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
    train_dataset = CropCXRDataset(
        train_df,
        train_ids,
        image_index,
        columns,
        original_size_map,
        int(cfg["image_size"]),
        float(cfg.get("crop_margin", 0.4)),
    )
    val_dataset = CropCXRDataset(
        train_df,
        val_ids,
        image_index,
        columns,
        original_size_map,
        int(cfg["image_size"]),
        float(cfg.get("crop_margin", 0.4)),
    )
    logger.info("Crop examples | train=%d val=%d", len(train_dataset), len(val_dataset))

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(cfg.get("batch_size", 32)),
        shuffle=True,
        num_workers=int(cfg.get("num_workers", 2)),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(cfg.get("batch_size", 32)),
        shuffle=False,
        num_workers=int(cfg.get("num_workers", 2)),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_accelerator(logger, "Crop classifier")
    model = build_resnet18_classifier(num_classes=14, pretrained=False).to(device)
    model = maybe_wrap_data_parallel(model, cfg, logger, "Crop classifier")
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg.get("lr", 3e-4)), weight_decay=float(cfg.get("weight_decay", 1e-4)))
    amp_enabled = bool(cfg.get("amp", True) and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    ckpt_path = Path(cfg["checkpoint"])
    best_loss = float("inf")
    if bool(cfg.get("resume", True)) and ckpt_path.exists():
        logger.info("Resuming crop classifier from %s", ckpt_path)
        checkpoint = load_checkpoint(ckpt_path, model, optimizer=optimizer, scaler=scaler, map_location=device)
        best_loss = float(checkpoint.get("best_score", best_loss))

    history = []
    epochs = 1 if bool(config.get("smoke_test", False)) else int(cfg.get("epochs", 3))
    for epoch in range(epochs):
        train_loss = train_multiclass_epoch(model, train_loader, optimizer, device, amp=amp_enabled, scaler=scaler)
        if len(val_dataset):
            val_loss, val_acc = evaluate_multiclass_loss(model, val_loader, device)
        else:
            val_loss, val_acc = train_loss, 0.0
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_acc": val_acc})
        logger.info("Crop epoch %d | train %.5f | val %.5f | acc %.4f | best %.5f", epoch, train_loss, val_loss, val_acc, best_loss)
        if val_loss < best_loss:
            best_loss = val_loss
            save_checkpoint(ckpt_path, model, optimizer, scaler, epoch, best_loss, config)

    summary_path = output_path(config, "lgcxr_crop_classifier_training_summary.json")
    summary_path.write_text(json.dumps({"best_loss": best_loss, "history": history}, indent=2), encoding="utf-8")
    logger.info("Crop classifier checkpoint: %s", ckpt_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
