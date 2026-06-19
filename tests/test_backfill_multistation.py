import pandas as pd

from infrastructure.kafka.scripts.backfill_multistation import build_training_frame, parse_open_meteo


def test_parse_open_meteo_to_met_by_hour():
    payload = {"hourly": {
        "time": ["2024-01-01T05:00", "2024-01-01T06:00"],
        "temperature_2m": [5.0, 6.0],
        "relativehumidity_2m": [60.0, 62.0],
        "dewpoint_2m": [1.0, 1.5],
        "windspeed_10m": [10.0, 11.0],
        "winddirection_10m": [270.0, 280.0],
    }}
    out = parse_open_meteo(payload, 43.64543, -79.38908)
    key = "2024-01-01 05:00:00+00:00"
    assert key in out
    rec = out[key][0]
    assert rec["temperature"] == 5.0
    assert rec["humidity"] == 60.0
    assert rec["dew_point"] == 1.0
    assert rec["wind_speed"] == 10.0
    assert rec["wind_dir"] == 270.0
    assert rec["latitude"] == 43.64543 and rec["longitude"] == -79.38908


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
    assert "timestamp" in df.columns           # training prep keys off "timestamp"


def test_build_training_frame_includes_target_gases():
    hour = "2024-01-01 05:00:00+00:00"
    per_station_pm = {
        7570:    pd.DataFrame({"datetime": [hour], "pm25": [14.0]}),
        1274950: pd.DataFrame({"datetime": [hour], "pm25": [20.0]}),
        1274949: pd.DataFrame({"datetime": [hour], "pm25": [10.0]}),
    }
    met_by_hour = {hour: [{"latitude": 43.70, "longitude": -79.40, "wind_speed": 10.0,
                           "wind_dir": 270.0, "temperature": 5.0, "dew_point": 1.0,
                           "humidity": 60.0, "pressure": 101.0}]}
    target_gases = {hour: {"no": 0.5, "no2": 1.2, "nox": 1.7, "o3": 30.0}}
    df = build_training_frame(per_station_pm, met_by_hour, target_gases=target_gases)
    assert df.iloc[0]["no"] == 0.5 and df.iloc[0]["no2"] == 1.2
    assert df.iloc[0]["nox"] == 1.7 and df.iloc[0]["o3"] == 30.0


def test_build_training_frame_gases_default_zero_when_absent():
    hour = "2024-01-01 05:00:00+00:00"
    per_station_pm = {7570: pd.DataFrame({"datetime": [hour], "pm25": [14.0]})}
    df = build_training_frame(per_station_pm, {}, target_gases=None)
    assert df.iloc[0]["no"] == 0.0 and df.iloc[0]["o3"] == 0.0
