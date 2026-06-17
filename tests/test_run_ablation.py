from scripts.run_ablation import RECIPES, flatten_results


def test_recipes_present():
    assert "base6" in RECIPES and "diffusion9" in RECIPES
    assert "upwind_pm25" in RECIPES["diffusion9"]
    assert "upwind_pm25" not in RECIPES["base6"]


def test_flatten_results_rows():
    rm = {1: {"Linear Regression": {
        "Train": {"R2": 0.9, "MAE": 1.0, "MSE": 2.0, "RMSE": 1.4},
        "Val":   {"R2": 0.8, "MAE": 1.1, "MSE": 2.1, "RMSE": 1.45},
        "Test":  {"R2": 0.7, "MAE": 1.2, "MSE": 2.2, "RMSE": 1.48},
    }}}
    rows = flatten_results(rm, "base6")
    test_row = [r for r in rows if r["split"] == "test"][0]
    assert test_row["feature_set"] == "base6"
    assert test_row["model"] == "Linear Regression"
    assert test_row["horizon"] == 1
    assert test_row["r2"] == 0.7 and test_row["rmse"] == 1.48
