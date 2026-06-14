"""Train/validation/test split helpers."""

from __future__ import annotations

import random

import pandas as pd

from .columns import DetectionColumns, extract_image_ids


def make_train_val_split(
    train_df: pd.DataFrame,
    columns: DetectionColumns,
    val_size: float,
    seed: int,
    all_image_ids: list[str] | None = None,
    smoke_test: bool = False,
    smoke_max_train_images: int = 500,
    smoke_max_val_images: int = 100,
) -> tuple[list[str], list[str]]:
    image_ids = ordered_unique([str(v) for v in all_image_ids] if all_image_ids is not None else extract_image_ids(train_df, columns.image_id))
    rng = random.Random(seed)
    rng.shuffle(image_ids)

    if len(image_ids) <= 1:
        return image_ids, []

    val_count = max(1, int(round(len(image_ids) * float(val_size))))
    val_count = min(val_count, len(image_ids) - 1)
    val_ids = image_ids[:val_count]
    train_ids = image_ids[val_count:]

    if smoke_test:
        train_ids = train_ids[: int(smoke_max_train_images)]
        val_ids = val_ids[: int(smoke_max_val_images)]

    return train_ids, val_ids


def sample_submission_image_ids(sample_df: pd.DataFrame, smoke_test: bool = False, smoke_max_test_images: int = 100) -> list[str]:
    if "image_id" not in sample_df.columns:
        raise ValueError("Sample submission must contain image_id")
    image_ids = ordered_unique([str(v) for v in sample_df["image_id"].astype(str).tolist()])
    if smoke_test:
        return image_ids[: int(smoke_max_test_images)]
    return image_ids


def ordered_unique(values) -> list[str]:
    seen = set()
    unique = []
    for value in values:
        text = str(value)
        if text in seen:
            continue
        seen.add(text)
        unique.append(text)
    return unique
