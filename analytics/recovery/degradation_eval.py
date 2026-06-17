import math
import random

from infrastructure.kafka.station_registry import target_id, neighbor_ids, coords
from .spatial_recovery import recover


def inject_missing(n, rate, seed):
    count = round(n * rate)
    rng = random.Random(seed)
    return sorted(rng.sample(range(n), count))


def evaluate_recovery(frame, rates, seed=0):
    t_lat, t_lon = coords(target_id())
    truth = [float(v) for v in frame["pm25"].tolist()]
    n = len(truth)
    out = {}
    for rate in rates:
        blanked = set(inject_missing(n, rate, seed))
        abs_errs = []
        sq_errs = []
        for i in blanked:
            neighbors = []
            for nid in neighbor_ids():
                col = f"pm25_{nid}"
                if col in frame.columns:
                    neighbors.append({"lat": coords(nid)[0], "lon": coords(nid)[1],
                                      "pm25": float(frame[col].iloc[i])})
            wu = float(frame["wind_u"].iloc[i]) if "wind_u" in frame.columns else 0.0
            wv = float(frame["wind_v"].iloc[i]) if "wind_v" in frame.columns else 0.0
            history = [truth[j] for j in range(max(0, i - 6), i)]
            value, _method = recover(t_lat, t_lon, wu, wv, neighbors, history, gap_hours=1)
            if value is None:
                continue
            err = value - truth[i]
            abs_errs.append(abs(err))
            sq_errs.append(err * err)
        count = len(abs_errs)
        out[rate] = {
            "mae": sum(abs_errs) / count if count else float("nan"),
            "rmse": math.sqrt(sum(sq_errs) / count) if count else float("nan"),
            "n": count,
        }
    return out


def main():  # pragma: no cover - I/O
    import argparse
    import pandas as pd
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default="data/external/multistation/train.csv")
    ap.add_argument("--rates", default="0.05,0.10,0.20")
    args = ap.parse_args()
    frame = pd.read_csv(args.path)
    rates = [float(x) for x in args.rates.split(",")]
    results = evaluate_recovery(frame, rates)
    for rate, m in results.items():
        print(f"missing={rate:.0%}  MAE={m['mae']:.3f}  RMSE={m['rmse']:.3f}  recovered={m['n']}")


if __name__ == "__main__":  # pragma: no cover
    main()
