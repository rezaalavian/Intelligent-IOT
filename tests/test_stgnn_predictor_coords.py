import pytest

pytest.importorskip("torch")
pytest.importorskip("torch_geometric")


def test_predictor_default_coords_are_registry():
    from models.predictors import STGNNPredictor, AirQualitySTGNN
    from infrastructure.kafka.station_registry import STATIONS
    import numpy as np
    model = AirQualitySTGNN(num_features=9, num_timesteps_input=12)

    class _S:  # minimal scaler stub
        def transform(self, x):
            return np.asarray(x, dtype=float)

    p = STGNNPredictor(model=model, scaler=_S(),
                       feature_columns=["pm25"] * 9, lookback=12)
    assert {round(v[0], 4) for v in p.station_coords.values()} == \
           {round(s.lat, 4) for s in STATIONS.values()}
