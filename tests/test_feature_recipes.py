from models.feature_recipes import RECIPES


def test_recipes_available_and_consistent():
    assert set(["base6", "diffusion9", "base+upwind", "base+transport", "base+alignment"]) <= set(RECIPES)
    assert "upwind_pm25" in RECIPES["diffusion9"]
    assert "upwind_pm25" not in RECIPES["base6"]
    assert RECIPES["base+upwind"] == RECIPES["base6"] + ["upwind_pm25"]


def test_pollutant_recipes():
    assert "with_pollutants" in RECIPES and "base+pollutants" in RECIPES
    for g in ("no", "no2", "nox", "o3"):
        assert g in RECIPES["with_pollutants"]
        assert g in RECIPES["base+pollutants"]
    assert "upwind_pm25" in RECIPES["with_pollutants"]      # diffusion + gases
    assert "upwind_pm25" not in RECIPES["base+pollutants"]  # base + gases only
