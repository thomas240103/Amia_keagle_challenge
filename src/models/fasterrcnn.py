"""Faster R-CNN model builder."""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from urllib.parse import urlparse

import torch
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def build_fasterrcnn(num_classes: int, image_size: int = 800, max_size: int | None = None):
    """Build torchvision Faster R-CNN and replace the classification head.

    `num_classes` includes the background class. For this competition V1 uses
    15 classes: background plus pathology IDs 0..13 mapped to labels 1..14.
    """
    import torchvision.models.detection as detection

    builder = getattr(detection, "fasterrcnn_resnet50_fpn_v2", None)
    weights_cls = getattr(detection, "FasterRCNN_ResNet50_FPN_V2_Weights", None)
    if builder is None:
        builder = getattr(detection, "fasterrcnn_resnet50_fpn")
        weights_cls = getattr(detection, "FasterRCNN_ResNet50_FPN_Weights", None)

    weights = _cached_or_allowed_weights(weights_cls)
    if max_size is None:
        max_size = int(round(image_size * 1.5))

    kwargs = {
        "weights": weights,
        "weights_backbone": None,
        "min_size": int(image_size),
        "max_size": int(max_size),
    }

    try:
        model = builder(**kwargs)
    except TypeError:
        model = builder(
            pretrained=weights is not None,
            pretrained_backbone=False,
            min_size=int(image_size),
            max_size=int(max_size),
        )

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def _cached_or_allowed_weights(weights_cls):
    if weights_cls is None:
        return None
    weights = weights_cls.DEFAULT
    if os.environ.get("LGCXR_ALLOW_WEIGHT_DOWNLOAD", "0") == "1":
        return weights

    filename = Path(urlparse(getattr(weights, "url", "")).path).name
    if not filename:
        return None
    cached = Path(torch.hub.get_dir()) / "checkpoints" / filename
    if cached.exists():
        return weights

    warnings.warn(
        "Torchvision pretrained detector weights were not found in the local cache. "
        "Using randomly initialized detector weights. Set LGCXR_ALLOW_WEIGHT_DOWNLOAD=1 "
        "to permit torchvision downloads when internet access is available.",
        RuntimeWarning,
    )
    return None
