"""Submission formatting and validation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.image_paths import image_id_variants


PREDICTION_COLUMNS = ["image_id", "class_id", "confidence", "xmin", "ymin", "xmax", "ymax"]
SUBMISSION_COLUMNS = ["image_id", "PredictionString"]


def make_submission(
    sample_submission: pd.DataFrame,
    predictions: pd.DataFrame,
    output_path: str | Path,
    no_finding_string: str = "14 1.0 0 0 1 1",
) -> pd.DataFrame:
    if list(sample_submission.columns) != SUBMISSION_COLUMNS:
        raise ValueError(f"Sample submission columns must be exactly {SUBMISSION_COLUMNS}")

    if predictions.empty:
        predictions = pd.DataFrame(columns=PREDICTION_COLUMNS)
    missing = [col for col in PREDICTION_COLUMNS if col not in predictions.columns]
    if missing:
        raise ValueError(f"Prediction CSV is missing columns: {missing}")

    predictions = predictions.copy()
    predictions["image_id"] = predictions["image_id"].astype(str)
    predictions = predictions.drop_duplicates(
        subset=["image_id", "class_id", "confidence", "xmin", "ymin", "xmax", "ymax"],
        keep="first",
    )
    grouped = {}
    for image_id, group in predictions.groupby("image_id", sort=False):
        sorted_group = group.sort_values("confidence", ascending=False)
        for variant in image_id_variants(str(image_id)):
            grouped.setdefault(variant, sorted_group)

    rows = []
    for image_id in sample_submission["image_id"].astype(str).tolist():
        group = _lookup_prediction_group(grouped, image_id)
        if group is None or len(group) == 0:
            prediction_string = no_finding_string
        else:
            parts = []
            for _, row in group.iterrows():
                class_id = int(row["class_id"])
                if class_id < 0 or class_id > 13:
                    continue
                parts.extend(
                    [
                        str(class_id),
                        _format_float(float(row["confidence"])),
                        _format_float(float(row["xmin"])),
                        _format_float(float(row["ymin"])),
                        _format_float(float(row["xmax"])),
                        _format_float(float(row["ymax"])),
                    ]
                )
            prediction_string = " ".join(parts) if parts else no_finding_string
        rows.append({"image_id": image_id, "PredictionString": prediction_string})

    submission = pd.DataFrame(rows, columns=SUBMISSION_COLUMNS)
    validate_submission(submission, sample_submission, no_finding_string)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)
    return submission


def validate_submission(
    submission: pd.DataFrame,
    sample_submission: pd.DataFrame,
    no_finding_string: str = "14 1.0 0 0 1 1",
) -> None:
    if list(submission.columns) != SUBMISSION_COLUMNS:
        raise AssertionError(f"Submission columns must be exactly {SUBMISSION_COLUMNS}")
    if len(submission) != len(sample_submission):
        raise AssertionError("Submission row count must equal sample submission row count")
    if submission.isna().any().any():
        raise AssertionError("Submission must not contain NaN values")

    for prediction_string in submission["PredictionString"].astype(str):
        if not prediction_string.strip():
            raise AssertionError("PredictionString must never be empty")
        tokens = prediction_string.split()
        if len(tokens) % 6 != 0:
            raise AssertionError(f"PredictionString group length must be a multiple of 6: {prediction_string}")

        class_ids = [int(float(tokens[i])) for i in range(0, len(tokens), 6)]
        if any(class_id < 0 or class_id > 14 for class_id in class_ids):
            raise AssertionError(f"Class IDs must be in 0..14: {prediction_string}")
        if 14 in class_ids and prediction_string != no_finding_string:
            raise AssertionError("Class 14 may only appear alone as the exact no-finding fallback")


def _lookup_prediction_group(grouped: dict, image_id: str):
    for variant in image_id_variants(str(image_id)):
        group = grouped.get(variant)
        if group is not None:
            return group
    return None


def _format_float(value: float) -> str:
    return f"{value:.6g}"
