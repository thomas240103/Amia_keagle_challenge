"""Dataset discovery helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .columns import infer_detection_columns
from .image_paths import list_image_files


@dataclass
class DatasetDiscovery:
    root: Path
    csv_paths: list[Path]
    train_csv: Path | None
    test_csv: Path | None
    img_size_csv: Path | None
    sample_submission: Path | None
    train_dir: Path | None
    test_dir: Path | None
    train_image_paths: list[Path]
    test_image_paths: list[Path]
    image_paths: list[Path]


def find_csv_files(root: str | Path) -> list[Path]:
    root = Path(root)
    if not root.exists():
        return []
    return sorted(root.rglob("*.csv"))


def load_csv(path: str | Path, **kwargs) -> pd.DataFrame:
    return pd.read_csv(path, **kwargs)


def identify_sample_submission(csv_paths: list[Path]) -> Path | None:
    scored: list[tuple[int, Path]] = []
    for path in csv_paths:
        score = 0
        name = path.name.lower()
        if "sample" in name and "submission" in name:
            score += 5
        elif "submission" in name:
            score += 2
        try:
            columns = [str(c).lower() for c in pd.read_csv(path, nrows=3).columns]
        except Exception:
            continue
        if "image_id" in columns and "predictionstring" in columns:
            score += 10
        if score > 0:
            scored.append((score, path))
    return sorted(scored, reverse=True)[0][1] if scored else None


def identify_train_csv(csv_paths: list[Path], sample_submission: Path | None = None) -> Path | None:
    scored: list[tuple[int, Path]] = []
    for path in csv_paths:
        if sample_submission and path.resolve() == sample_submission.resolve():
            continue
        score = 0
        name = path.name.lower()
        if "train" in name:
            score += 4
        if "annotation" in name or "bbox" in name or "label" in name:
            score += 2
        try:
            sample = pd.read_csv(path, nrows=100)
            infer_detection_columns(sample)
            score += 10
        except Exception:
            continue
        scored.append((score, path))
    return sorted(scored, reverse=True)[0][1] if scored else None


def _existing_file(root: Path, name: str) -> Path | None:
    path = root / name
    return path if path.exists() and path.is_file() else None


def _existing_dir(root: Path, name: str) -> Path | None:
    path = root / name
    return path if path.exists() and path.is_dir() else None


def discover_dataset(root: str | Path) -> DatasetDiscovery:
    root = Path(root)
    csv_paths = find_csv_files(root)
    sample_submission = _existing_file(root, "sample_submission.csv") or identify_sample_submission(csv_paths)
    train_csv = _existing_file(root, "train.csv") or identify_train_csv(csv_paths, sample_submission)
    test_csv = _existing_file(root, "test.csv")
    img_size_csv = _existing_file(root, "img_size.csv")
    train_dir = _existing_dir(root, "train")
    test_dir = _existing_dir(root, "test")
    train_image_paths = list_image_files(train_dir) if train_dir else []
    test_image_paths = list_image_files(test_dir) if test_dir else []
    image_paths = list_image_files(root)
    return DatasetDiscovery(
        root=root,
        csv_paths=csv_paths,
        train_csv=train_csv,
        test_csv=test_csv,
        img_size_csv=img_size_csv,
        sample_submission=sample_submission,
        train_dir=train_dir,
        test_dir=test_dir,
        train_image_paths=train_image_paths,
        test_image_paths=test_image_paths,
        image_paths=image_paths,
    )
