from analytics.features.diffusion_features import diffusion_features

TARGET = (43.64543, -79.38908)  # Downtown


def test_single_neighbor_upwind_equals_its_pm25():
    # One neighbor with pm25 -> weighted average over one weight = that pm25
    n = [{"lat": 43.709444, "lon": -79.5435, "pm25": 20.0}]
    out = diffusion_features(*TARGET, wind_u=1.0, wind_v=0.0, neighbors=n)
    assert round(out["upwind_pm25"], 6) == 20.0


def test_missing_pm25_neighbor_dropped():
    n = [{"lat": 43.709444, "lon": -79.5435, "pm25": None}]
    out = diffusion_features(*TARGET, wind_u=1.0, wind_v=0.0, neighbors=n)
    assert out == {"upwind_pm25": 0.0, "transport_potential": 0.0, "wind_alignment": 0.0}


def test_alignment_in_range_and_keys():
    n = [{"lat": 43.78043, "lon": -79.467397, "pm25": 10.0},
         {"lat": 43.7453, "lon": -79.2703, "pm25": 30.0}]
    out = diffusion_features(*TARGET, wind_u=2.0, wind_v=1.0, neighbors=n)
    assert set(out) == {"upwind_pm25", "transport_potential", "wind_alignment"}
    assert -1.0 <= out["wind_alignment"] <= 1.0
