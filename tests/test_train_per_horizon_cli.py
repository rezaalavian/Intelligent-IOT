from scripts.train_per_horizon import parse_map
from models.feature_recipes import RECIPES


def test_parse_map_resolves_recipe_names():
    out = parse_map("1=base6,2=diffusion9,3=diffusion9", RECIPES)
    assert out[1] == RECIPES["base6"]
    assert out[2] == RECIPES["diffusion9"]
    assert set(out) == {1, 2, 3}


def test_parse_map_rejects_unknown_recipe():
    import pytest
    with pytest.raises(KeyError):
        parse_map("1=nope", RECIPES)
