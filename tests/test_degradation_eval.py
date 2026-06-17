import pandas as pd

from analytics.recovery.degradation_eval import inject_missing, evaluate_recovery


def test_inject_missing_deterministic_count():
    idx = inject_missing(100, 0.10, seed=1)
    assert len(idx) == 10
    assert inject_missing(100, 0.10, seed=1) == idx  # deterministic


def test_evaluate_recovery_reports_metrics():
    # target 7570 + neighbor 1274950 columns; identical values -> perfect spatial recovery
    df = pd.DataFrame({
        "pm25": [10.0, 12.0, 14.0, 11.0, 13.0, 9.0, 15.0, 8.0, 10.0, 12.0],
        "pm25_1274950": [10.0, 12.0, 14.0, 11.0, 13.0, 9.0, 15.0, 8.0, 10.0, 12.0],
        "wind_u": [1.0] * 10,
        "wind_v": [0.0] * 10,
    })
    out = evaluate_recovery(df, rates=[0.2], seed=1)
    assert 0.2 in out
    assert "mae" in out[0.2] and "rmse" in out[0.2]
    assert out[0.2]["mae"] < 1.0  # near-perfect since neighbor equals target
