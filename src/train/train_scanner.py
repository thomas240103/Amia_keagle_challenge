"""Training helpers for the Faster R-CNN scanner."""

from __future__ import annotations

from typing import Iterable

import torch
from tqdm.auto import tqdm

from src.utils.metrics_voc import voc_map


def detection_collate(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)


def move_targets_to_device(targets: Iterable[dict], device: torch.device) -> list[dict]:
    moved = []
    for target in targets:
        moved.append({key: value.to(device) if torch.is_tensor(value) else value for key, value in target.items()})
    return moved


def make_optimizer(model: torch.nn.Module, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.AdamW(params, lr=float(lr), weight_decay=float(weight_decay))


def train_one_epoch(
    model: torch.nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    amp: bool = True,
    grad_clip: float | None = None,
    scaler=None,
    epoch: int = 0,
) -> float:
    model.train()
    losses = []
    amp_enabled = bool(amp and device.type == "cuda")
    progress = tqdm(loader, desc=f"train epoch {epoch}", leave=False)

    for images, targets in progress:
        images = [image.to(device) for image in images]
        targets = move_targets_to_device(targets, device)
        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=amp_enabled):
            loss_dict = model(images, targets)
            loss = sum(loss for loss in loss_dict.values())

        if scaler is not None and amp_enabled:
            scaler.scale(loss).backward()
            if grad_clip:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
            optimizer.step()

        loss_value = float(loss.detach().cpu().item())
        losses.append(loss_value)
        progress.set_postfix(loss=f"{loss_value:.4f}")

    return float(sum(losses) / max(len(losses), 1))


@torch.no_grad()
def evaluate_detector_map(
    model: torch.nn.Module,
    loader,
    dataset,
    device: torch.device,
    iou_threshold: float = 0.4,
    conf_threshold: float = 0.0,
) -> dict:
    model.eval()
    ground_truth = []
    predictions = []

    for images, targets in tqdm(loader, desc="valid", leave=False):
        images = [image.to(device) for image in images]
        outputs = model(images)
        for output, target in zip(outputs, targets):
            image_idx = int(target["image_id"].item())
            image_id = dataset.image_ids[image_idx]

            gt_boxes = target["boxes"].detach().cpu().numpy()
            gt_labels = target["labels"].detach().cpu().numpy()
            for box, label in zip(gt_boxes, gt_labels):
                if 1 <= int(label) <= 14:
                    ground_truth.append(
                        {"image_id": image_id, "class_id": int(label) - 1, "box": [float(v) for v in box]}
                    )

            boxes = output["boxes"].detach().cpu().numpy()
            labels = output["labels"].detach().cpu().numpy()
            scores = output["scores"].detach().cpu().numpy()
            for box, label, score in zip(boxes, labels, scores):
                if float(score) < conf_threshold or int(label) < 1 or int(label) > 14:
                    continue
                predictions.append(
                    {
                        "image_id": image_id,
                        "class_id": int(label) - 1,
                        "score": float(score),
                        "box": [float(v) for v in box],
                    }
                )

    return voc_map(ground_truth, predictions, class_ids=range(14), iou_threshold=float(iou_threshold))
