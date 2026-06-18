from models.feature_recipes import RECIPES


def test_recipes_available_and_consistent():
    assert set(["base6", "diffusion9", "base+upwind", "base+transport", "base+alignment"]) <= set(RECIPES)
    assert "upwind_pm25" in RECIPES["diffusion9"]
    assert "upwind_pm25" not in RECIPES["base6"]
    assert RECIPES["base+upwind"] == RECIPES["base6"] + ["upwind_pm25"]
