from infrastructure.kafka.met_join import nearest_met


def test_picks_closest():
    recs = [
        {"latitude": 43.70, "longitude": -79.40, "wind_speed": 5.0},
        {"latitude": 44.50, "longitude": -79.00, "wind_speed": 9.0},
    ]
    out = nearest_met(43.645, -79.389, recs)
    assert out["wind_speed"] == 5.0


def test_empty_returns_none():
    assert nearest_met(43.0, -79.0, []) is None
