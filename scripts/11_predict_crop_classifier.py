#!/usr/bin/env python
"""Predict optional V2 crop classifier verification scores for scanner boxes."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.classification_dataset import expand_box, normalize_tensor
from src.data.discovery import discover_dataset
from src.data.image_paths import build_image_index, resolve_image_path
from src.data.image_sizes import load_image_size_map, scale_original_boxes_to_png
from src.models.resnet18_classifier import build_resnet18_classifier
from src.utils.boxes import clip_boxes_to_image
from src.utils.checkpoints import load_checkpoint
from src.utils.config import apply_runtime_overrides, ensure_work_dir, load_config, output_path
from src.utils.logging import configure_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict crop classifier scores.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--work-dir", default=None)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--scanner-predictions", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logger = configure_logger()
    config = apply_runtime_overrides(load_config(args.config), args.data_root, args.work_dir, args.smoke_test)
    ensure_work_dir(config)
    cfg = config["crop_classifier"]
    if not bool(cfg.get("enabled", False)):
        logger.info("Crop classifier disabled; skipping prediction.")
        return 0
    ckpt_path = Path(cfg["checkpoint"])
    if not ckpt_path.exists():
        logger.warning("Crop classifier checkpoint missing; skipping: %s", ckpt_path)
        return 0

    scanner_path = Path(args.scanner_predictions) if args.scanner_predictions else output_path(config, "lgcxr_fast_test_predictions.csv")
    if not scanner_path.exists():
        raise FileNotFoundError(f"Scanner predictions not found: {scanner_path}")
    scanner_df = pd.read_csv(scanner_path).reset_index().rename(columns={"index": "prediction_id"})
    if scanner_df.empty:
        output_path(config, "lgcxr_crop_test_scores.csv").write_text("prediction_id,image_id,class_id,crop_score\n", encoding="utf-8")
        return 0

    discovery = discover_dataset(config["data_root"])
    image_index = build_image_index(discovery.test_image_paths or discovery.image_paths)
    original_size_map, _ = load_image_size_map(discovery.img_size_csv)
    dataset = CropPredictionDataset(scanner_df, image_index, original_size_map, int(cfg["image_size"]), float(cfg.get("crop_margin", 0.4)))
    loader = DataLoader(dataset, batch_size=int(cfg.get("batch_size", 32)), shuffle=False, num_workers=int(cfg.get("num_workers", 2)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_resnet18_classifier(num_classes=14, pretrained=False).to(device)
    load_checkpoint(ckpt_path, model, map_location=device)
    model.eval()

    records = []
    with torch.no_grad():
        for images, class_ids, prediction_ids, image_ids in loader:
            probs = torch.softmax(model(images.to(device)), dim=1).cpu()
            for prob, class_id, prediction_id, image_id in zip(probs, class_ids, prediction_ids, image_ids):
                cid = int(class_id.item())
                records.append(
                    {
                        "prediction_id": int(prediction_id.item()),
                        "image_id": str(image_id),
                        "class_id": cid,
                        "crop_score": float(prob[cid].item()),
                    }
                )

    out = output_path(config, "lgcxr_crop_test_scores.csv")
    pd.DataFrame(records).to_csv(out, index=False)
    logger.info("Crop scores saved to %s", out)
    return 0


class CropPredictionDataset(Dataset):
    def __init__(self, predictions, image_index, original_size_map, image_size: int, crop_margin: float) -> None:
        self.predictions = predictions.to_dict("records")
        self.image_index = image_index
        self.original_size_map = original_size_map
        self.image_size = int(image_size)
        self.crop_margin = float(crop_margin)

    def __len__(self) -> int:
        return len(self.predictions)

    def __getitem__(self, idx: int):
        row = self.predictions[idx]
        image_id = str(row["image_id"])
        path = resolve_image_path(image_id, self.image_index)
        if path is None:
            raise FileNotFoundError(f"Could not resolve image path for image_id={image_id}")
        image = Image.open(path).convert("RGB")
        width, height = image.size
        box = np.asarray([[row["xmin"], row["ymin"], row["xmax"], row["ymax"]]], dtype=float)
        box, _ = scale_original_boxes_to_png(box, image_id, self.original_size_map, width, height)
        box = clip_boxes_to_image(expand_box(box[0], self.crop_margin), width, height)[0]
        crop = image.crop(tuple(float(v) for v in box)).resize((self.image_size, self.image_size), Image.BILINEAR)
        tensor = normalize_tensor(F.to_tensor(crop))
        class_id = torch.tensor(int(row["class_id"]), dtype=torch.long)
        prediction_id = torch.tensor(int(row["prediction_id"]), dtype=torch.long)
        return tensor, class_id, prediction_id, image_id


if __name__ == "__main__":
    raise SystemExit(main())
