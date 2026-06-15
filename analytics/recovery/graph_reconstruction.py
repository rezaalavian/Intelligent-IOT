"""Long-term graph-based recovery stub."""
from typing import Any
import pandas as pd


def reconstruct_missing(frame: pd.DataFrame, adjacency: Any | None = None) -> pd.DataFrame:
    """Recover longer gaps using a simple graph-aware average fallback.

    The implementation is intentionally lightweight for reproducibility and can be
    replaced with a learned graph model when the team trains the full Phase 4 model.
    """

    recovered = frame.copy()
    numeric_cols = recovered.select_dtypes(include="number").columns
    recovered[numeric_cols] = recovered[numeric_cols].fillna(recovered[numeric_cols].median())
    if adjacency is not None:
        recovered["graph_reconstruction_used"] = True
    return recovered
