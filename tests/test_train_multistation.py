from models.baselines import train_baselines as tb


def test_feature_cols_include_diffusion():
    for col in ["upwind_pm25", "transport_potential", "wind_alignment"]:
        assert col in tb.FEATURE_COLS


def test_station_coords_from_registry():
    from infrastructure.kafka.station_registry import STATIONS
    assert set(tb.STATION_COORDS.keys()) == set(STATIONS.keys())
