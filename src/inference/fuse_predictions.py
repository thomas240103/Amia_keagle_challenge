"""Prediction fusion helpers."""

from __future__ import annotations

import pandas as pd

from src.data.image_paths import image_id_variants


def fuse_v1(scanner_predictions: pd.DataFrame, *_, **__) -> pd.DataFrame:
    """Return scanner predictions unchanged for V1."""
    return scanner_predictions.copy()


def fuse_three_model_scores(
    scanner_predictions: pd.DataFrame,
    global_scores: pd.DataFrame | None,
    crop_scores: pd.DataFrame | None,
    scanner_weight: float = 0.7,
    global_weight: float = 0.15,
    crop_weight: float = 0.15,
    default_threshold: float = 0.03,
    per_class_thresholds: dict | None = None,
) -> pd.DataFrame:
    """Fuse scanner confidence with optional global/crop classifier scores."""
    if scanner_predictions.empty:
        return scanner_predictions.copy()

    df = scanner_predictions.copy().reset_index(drop=True)
    df["prediction_id"] = range(len(df))
    df["_canonical_image_id"] = df["image_id"].map(canonical_image_id)
    df["class_id"] = df["class_id"].astype(int)
    df["scanner_score"] = df["confidence"].astype(float)

    if global_scores is not None and not global_scores.empty:
        g = global_scores.copy()
        g["_canonical_image_id"] = g["image_id"].map(canonical_image_id)
        g["class_id"] = g["class_id"].astype(int)
        df = df.merge(
            g[["_canonical_image_id", "class_id", "global_score"]],
            on=["_canonical_image_id", "class_id"],
            how="left",
        )
    else:
        df["global_score"] = pd.NA

    if crop_scores is not None and not crop_scores.empty:
        c = crop_scores.copy()
        c["prediction_id"] = c["prediction_id"].astype(int)
        df = df.merge(c[["prediction_id", "crop_score"]], on="prediction_id", how="left")
    else:
        df["crop_score"] = pd.NA

    fused = []
    for _, row in df.iterrows():
        total = float(scanner_weight)
        score = float(scanner_weight) * float(row["scanner_score"])
        if pd.notna(row.get("global_score")) and float(global_weight) > 0:
            total += float(global_weight)
            score += float(global_weight) * float(row["global_score"])
        if pd.notna(row.get("crop_score")) and float(crop_weight) > 0:
            total += float(crop_weight)
            score += float(crop_weight) * float(row["crop_score"])
        fused.append(score / max(total, 1e-9))
    df["confidence"] = fused

    thresholds = per_class_thresholds or {}
    keep = []
    for _, row in df.iterrows():
        threshold = float(thresholds.get(str(int(row["class_id"])), thresholds.get(int(row["class_id"]), default_threshold)))
        keep.append(float(row["confidence"]) >= threshold)
    df = df[keep].copy()
    df = df.sort_values(["image_id", "confidence"], ascending=[True, False])
    return df[["image_id", "class_id", "confidence", "xmin", "ymin", "xmax", "ymax"]]


def canonical_image_id(value: str) -> str:
    variants = sorted(image_id_variants(str(value)), key=len)
    return variants[0] if variants else str(value).lower()
