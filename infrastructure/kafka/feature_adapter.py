import math

MODEL_FEATURE_KEYS = [
    "temp definition °c",
    "dew point definition °c",
    "rel hum definition %",
    "wind_u",
    "wind_v",
    "pm25",
]


def _f(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def to_model_features(measurement: dict) -> dict:
    speed = _f(measurement.get("wind_speed"))
    direction = math.radians(_f(measurement.get("wind_dir")))
    return {
        "temp definition °c": _f(measurement.get("temperature")),
        "dew point definition °c": _f(measurement.get("dew_point")),
        "rel hum definition %": _f(measurement.get("humidity")),
        "wind_u": speed * math.cos(direction),
        "wind_v": speed * math.sin(direction),
        "pm25": _f(measurement.get("pm25")),
    }
