"""Quick verification that active STGNN model loads and serves predictions."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from infrastructure.deployment.controller import load_controller


def main() -> int:
    controller = load_controller()
    if not controller.is_ready():
        print("FAIL: active_model.pkl not loaded. Run: python scripts/save_deployment_models.py")
        return 1

    features = {
        "temp definition °c": 5.0,
        "dew point definition °c": 0.0,
        "rel hum definition %": 70.0,
        "wind_u": 2.0,
        "wind_v": 1.0,
        "pm2": 35.0,
    }
    history = [dict(features) for _ in range(controller.bundle.lookback if controller.bundle else 12)]
    forecast = controller.predict({"features": features, "history": history})
    alert = controller.evaluate_alerts({"pm25": 80.0, **forecast})
    print("status:", controller.status())
    print("forecast:", forecast)
    print("alert:", alert)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
