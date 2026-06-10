from infrastructure.kafka.data_sources import environment_canada as ec


SAMPLE_FEATURE = {
    "geometry": {"coordinates": [-79.4, 43.7, 100.0]},
    "properties": {
        "stn_nam-value": "TORONTO",
        "clim_id-value": "6158359",
        "date_tm-value": "2023-01-01T14:00:00Z",
        "air_temp-value": "-3.2", "air_temp-qa": "100",
        "rel_hum-value": "81", "rel_hum-qa": "100",
        "avg_wnd_spd_10m_pst1mt-value": "12.0", "avg_wnd_spd_10m_pst1mt-qa": "100",
        "avg_wnd_dir_10m_pst1mt-value": "270", "avg_wnd_dir_10m_pst1mt-qa": "100",
        "stn_pres-value": "101.3", "stn_pres-qa": "100",
    },
}


def test_feature_to_raw_maps_swob_fields():
    rec = ec._feature_to_raw(SAMPLE_FEATURE)
    assert rec["station_id"] == "swob-6158359"
    assert rec["air_temp"] == -3.2
    assert rec["rel_hum"] == 81.0
    assert rec["wind_speed"] == 12.0
    assert rec["wind_dir"] == 270.0
    assert rec["pressure"] == 101.3
    assert rec["latitude"] == 43.7 and rec["longitude"] == -79.4
    assert rec["datetime_utc"] == "2023-01-01T14:00:00Z"


def test_qa_failure_nulls_the_value():
    feat = {"geometry": {"coordinates": [-79.4, 43.7]},
            "properties": {"clim_id-value": "X", "date_tm-value": "t",
                           "air_temp-value": "-3.2", "air_temp-qa": "10"}}
    rec = ec._feature_to_raw(feat, min_qa=50)
    assert rec["air_temp"] is None  # qa below threshold -> dropped


def test_collection_to_raw_iterates_features():
    coll = {"features": [SAMPLE_FEATURE, SAMPLE_FEATURE]}
    recs = ec._collection_to_raw(coll)
    assert len(recs) == 2
