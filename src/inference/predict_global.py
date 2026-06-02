"""Optional global classifier inference placeholder."""

from __future__ import annotations

from pathlib import Path
import warnings


def load_optional_global_classifier(checkpoint_path: str | Path):
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        warnings.warn(f"Global classifier checkpoint missing; continuing without it: {checkpoint_path}")
        return None
    raise NotImplementedError("Global classifier inference is intentionally deferred for V1.")
