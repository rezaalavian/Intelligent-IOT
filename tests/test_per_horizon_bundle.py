from models.forecast_bundle import ForecastBundle


class _EchoWidth:
    """A stub model whose prediction is the number of features it received."""
    def predict(self, X):
        return [float(X.shape[1])]


class _Identity:
    def transform(self, X):
        return X


def _bundle():
    return ForecastBundle(
        feature_columns=["a", "b"],
        target_column="pm25",
        horizons=[1, 2],
        model_type="per_horizon:test",
        scaler=_Identity(),
        models={1: _EchoWidth(), 2: _EchoWidth()},
        scalers={1: _Identity(), 2: _Identity()},
        feature_columns_by_horizon={1: ["a", "b"], 2: ["a", "b", "c"]},
    )


def test_each_horizon_uses_its_own_feature_set():
    b = _bundle()
    feats = {"a": 1.0, "b": 2.0, "c": 3.0}
    assert b.predict_horizon(feats, 1) == 2.0   # h1 -> 2 features
    assert b.predict_horizon(feats, 2) == 3.0   # h2 -> 3 features


def test_falls_back_to_single_list_when_no_per_horizon():
    b = _bundle()
    b.feature_columns_by_horizon = None
    feats = {"a": 1.0, "b": 2.0, "c": 3.0}
    assert b.predict_horizon(feats, 1) == 2.0   # uses feature_columns (["a","b"])
