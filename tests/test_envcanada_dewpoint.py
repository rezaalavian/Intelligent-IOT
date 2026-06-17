from infrastructure.kafka.data_sources.environment_canada import _feature_to_raw


def test_feature_to_raw_extracts_dew_point():
    feature = {
        "geometry": {"coordinates": [-79.4, 43.7]},
        "properties": {
            "clim_id-value": "6158359",
            "date_tm-value": "2026-06-16T14:00:00Z",
            "air_temp-value": 21.0,
            "dwpt_temp-value": 12.5,
        },
    }
    rec = _feature_to_raw(feature)
    assert rec["dew_point"] == 12.5
