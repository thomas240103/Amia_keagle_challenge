"""V1 prediction fusion placeholder."""

from __future__ import annotations

import pandas as pd


def fuse_v1(scanner_predictions: pd.DataFrame, *_, **__) -> pd.DataFrame:
    """Return scanner predictions unchanged for V1."""
    return scanner_predictions.copy()
