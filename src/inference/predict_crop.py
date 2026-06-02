"""Optional crop classifier inference placeholder."""

from __future__ import annotations

from pathlib import Path
import warnings


def load_optional_crop_classifier(checkpoint_path: str | Path):
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        warnings.warn(f"Crop classifier checkpoint missing; continuing without it: {checkpoint_path}")
        return None
    raise NotImplementedError("Crop classifier inference is intentionally deferred for V1.")
