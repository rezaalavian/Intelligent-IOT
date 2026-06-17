import pandas as pd

from infrastructure.kafka.scripts.backfill_multistation import build_training_frame


def test_build_training_frame_has_diffusion_and_target():
    hour = "2024-01-01 05:00:00+00:00"
    per_station_pm = {
        7570:    pd.DataFrame({"datetime": [hour], "pm25": [14.0]}),
        1274950: pd.DataFrame({"datetime": [hour], "pm25": [20.0]}),
        1274949: pd.DataFrame({"datetime": [hour], "pm25": [10.0]}),
        1210341: pd.DataFrame({"datetime": [hour], "pm25": [30.0]}),
    }
    met_by_hour = {hour: [
        {"latitude": 43.70, "longitude": -79.40, "wind_speed": 10.0, "wind_dir": 270.0,
         "temperature": 5.0, "dew_point": 1.0, "humidity": 60.0, "pressure": 101.0},
    ]}
    df = build_training_frame(per_station_pm, met_by_hour)
    assert len(df) == 1
    for col in ["temp definition °c", "dew point definition °c", "rel hum definition %",
                "wind_u", "wind_v", "pm25", "upwind_pm25", "transport_potential",
                "wind_alignment"]:
        assert col in df.columns
    assert df.iloc[0]["pm25"] == 14.0          # target station's pm25
    assert df.iloc[0]["upwind_pm25"] != 0.0    # neighbors contributed
