"""Optional ResNet18 classifier builder for future global/crop models."""

from __future__ import annotations

import torch


def build_resnet18_classifier(num_classes: int, pretrained: bool = False) -> torch.nn.Module:
    try:
        import timm

        return timm.create_model("resnet18", pretrained=pretrained, num_classes=num_classes)
    except Exception:
        from torchvision import models

        weights = models.ResNet18_Weights.DEFAULT if pretrained and hasattr(models, "ResNet18_Weights") else None
        model = models.resnet18(weights=weights)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
        return model
