from analytics.recovery.spatial_recovery import (
    wind_weighted_estimate, temporal_fallback, recover,
)

TARGET = (43.64543, -79.38908)


def test_single_neighbor_estimate_equals_its_pm25():
    n = [{"lat": 43.709444, "lon": -79.5435, "pm25": 18.0}]
    assert round(wind_weighted_estimate(*TARGET, 1.0, 0.0, n), 6) == 18.0


def test_estimate_none_when_no_neighbor_pm25():
    n = [{"lat": 43.709444, "lon": -79.5435, "pm25": None}]
    assert wind_weighted_estimate(*TARGET, 1.0, 0.0, n) is None


def test_temporal_fallback_last_non_null():
    assert temporal_fallback([5.0, None, 7.0, None]) == 7.0
    assert temporal_fallback([None, None]) is None
    assert temporal_fallback([]) is None


def test_recover_short_gap_uses_spatial():
    n = [{"lat": 43.709444, "lon": -79.5435, "pm25": 20.0}]
    value, method = recover(*TARGET, 1.0, 0.0, n, [9.0], gap_hours=1)
    assert method == "spatial" and round(value, 6) == 20.0


def test_recover_long_gap_uses_temporal():
    n = [{"lat": 43.709444, "lon": -79.5435, "pm25": 20.0}]
    value, method = recover(*TARGET, 1.0, 0.0, n, [9.0], gap_hours=10)
    assert method == "temporal" and value == 9.0


def test_recover_nothing_available():
    value, method = recover(*TARGET, 1.0, 0.0, [], [], gap_hours=1)
    assert value is None and method == "none"
