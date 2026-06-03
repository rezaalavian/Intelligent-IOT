"""Short-term spatial interpolation and recovery."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


def spatial_interpolate(frame: pd.DataFrame, value_columns: Sequence[str], group_column: str = "city_name") -> pd.DataFrame:
    """Fill short gaps using grouped forward/backward fill and rolling median.

    This is a practical first-stage recovery method that can be upgraded to kriging
    when a denser station network is available.
    """

    recovered = frame.copy()
    recovered = recovered.sort_values([group_column, "timestamp"] if group_column in recovered.columns else ["timestamp"])
    for column in value_columns:
        if column not in recovered.columns:
            continue
        recovered[column] = pd.to_numeric(recovered[column], errors="coerce")
        if group_column in recovered.columns:
            recovered[column] = (
                recovered.groupby(group_column)[column]
                .transform(lambda s: s.interpolate(limit_direction="both").fillna(s.median()))
            )
        else:
            recovered[column] = recovered[column].interpolate(limit_direction="both").fillna(recovered[column].median())
    return recovered
