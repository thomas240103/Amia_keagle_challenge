"""Checkpoint save/load helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from src.utils.accelerator import unwrap_parallel_model


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scaler: Any,
    epoch: int,
    best_score: float,
    config: dict,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model_to_save = unwrap_parallel_model(model)
    payload = {
        "model": model_to_save.state_dict(),
        "epoch": epoch,
        "best_score": best_score,
        "config": config,
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if scaler is not None:
        payload["scaler"] = scaler.state_dict()
    torch.save(payload, path)


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: Any = None,
    map_location: str | torch.device = "cpu",
) -> dict:
    checkpoint = torch.load(path, map_location=map_location)
    state_dict = checkpoint.get("model", checkpoint)
    model_to_load = unwrap_parallel_model(model)
    try:
        model_to_load.load_state_dict(state_dict)
    except RuntimeError:
        if _looks_like_data_parallel_state(state_dict):
            model_to_load.load_state_dict(_strip_data_parallel_prefix(state_dict))
        else:
            raise
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scaler is not None and "scaler" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler"])
    return checkpoint


def _looks_like_data_parallel_state(state_dict: dict) -> bool:
    return any(str(key).startswith("module.") for key in state_dict)


def _strip_data_parallel_prefix(state_dict: dict) -> dict:
    return {str(key).removeprefix("module."): value for key, value in state_dict.items()}
