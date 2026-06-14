"""CUDA accelerator helpers."""

from __future__ import annotations

from typing import Any

import torch


def cuda_device_count() -> int:
    if not torch.cuda.is_available():
        return 0
    return int(torch.cuda.device_count())


def cuda_device_names() -> list[str]:
    return [str(torch.cuda.get_device_name(index)) for index in range(cuda_device_count())]


def log_accelerator(logger: Any, component: str) -> int:
    count = cuda_device_count()
    if count == 0:
        logger.info("%s accelerator: CUDA unavailable; using CPU.", component)
        return 0

    names = ", ".join(f"{index}:{name}" for index, name in enumerate(cuda_device_names()))
    logger.info("%s accelerator: CUDA available with %d GPU(s): %s", component, count, names)
    return count


def maybe_wrap_data_parallel(model: torch.nn.Module, cfg: dict, logger: Any, component: str) -> torch.nn.Module:
    """Use all visible GPUs for classifier models when configured.

    DataParallel keeps the same model/loss/output semantics. The config controls
    whether it is used, and "auto" enables it only when more than one CUDA GPU is
    visible.
    """

    count = cuda_device_count()
    setting = cfg.get("multi_gpu", "auto")
    enabled = _multi_gpu_enabled(setting, count)
    if enabled and count > 1:
        logger.info("%s multi-GPU: enabled with torch.nn.DataParallel over %d GPU(s).", component, count)
        return torch.nn.DataParallel(model)

    if count > 1:
        logger.info("%s multi-GPU: disabled by config; using cuda:0 only.", component)
    return model


def unwrap_parallel_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, torch.nn.DataParallel) else model


def _multi_gpu_enabled(setting: object, count: int) -> bool:
    if isinstance(setting, bool):
        return setting
    if setting is None:
        return False
    value = str(setting).strip().lower()
    if value in {"auto", "true", "yes", "1", "on"}:
        return count > 1 if value == "auto" else True
    if value in {"false", "no", "0", "off", "none"}:
        return False
    return False
