from models.baselines.train_baselines import select_feature_cols, FEATURE_COLS


def test_none_uses_default_intersected_with_available():
    avail = ["temp definition °c", "pm25", "wind_u", "unrelated"]
    out = select_feature_cols(None, avail)
    assert out == [c for c in FEATURE_COLS if c in avail]


def test_explicit_subset_filtered_and_ordered():
    avail = ["pm25", "upwind_pm25", "wind_u", "temp definition °c"]
    req = ["temp definition °c", "pm25", "missing_col", "upwind_pm25"]
    out = select_feature_cols(req, avail)
    assert out == ["temp definition °c", "pm25", "upwind_pm25"]  # order preserved, missing dropped
