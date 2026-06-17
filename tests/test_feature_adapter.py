import math

from infrastructure.kafka.feature_adapter import to_model_features, MODEL_FEATURE_KEYS


def test_maps_canonical_names_to_model_keys():
    rec = {"temperature": 20.0, "dew_point": 12.0, "humidity": 55.0,
           "wind_speed": 10.0, "wind_dir": 0.0, "pm25": 14.0}
    feat = to_model_features(rec)
    assert feat["temp definition °c"] == 20.0
    assert feat["dew point definition °c"] == 12.0
    assert feat["rel hum definition %"] == 55.0
    assert feat["pm25"] == 14.0


def test_wind_components_from_degrees():
    rec = {"wind_speed": 10.0, "wind_dir": 90.0}
    feat = to_model_features(rec)
    assert feat["wind_u"] == math.cos(math.radians(90.0)) * 10.0
    assert feat["wind_v"] == math.sin(math.radians(90.0)) * 10.0


def test_missing_fields_default_to_zero():
    feat = to_model_features({})
    assert set(feat.keys()) == set(MODEL_FEATURE_KEYS)
    assert all(v == 0.0 for v in feat.values())
