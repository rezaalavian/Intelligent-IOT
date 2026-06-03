"""Normalize timestamp column in merged historical CSV produced by merge_openaq_into_historical.py

If the merged CSV has `timestamp_x`/`timestamp_y` instead of `timestamp`, this script
creates a canonical `timestamp` column and writes a fixed CSV.
"""
from pathlib import Path
import pandas as pd


def fix(input_path: str | Path = "data/raw/historical_rawdata_with_openaq.csv", output_path: str | Path = None):
    inp = Path(input_path)
    if output_path is None:
        output_path = inp.with_name(inp.stem + "_fixed" + inp.suffix)
    out = Path(output_path)

    df = pd.read_csv(inp, low_memory=False)

    if "timestamp" not in df.columns:
        if "timestamp_x" in df.columns:
            df["timestamp"] = df["timestamp_x"]
        elif "timestamp_y" in df.columns:
            df["timestamp"] = df["timestamp_y"]
        else:
            raise RuntimeError("No timestamp column found to normalize")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    # drop helper columns if present
    for c in ["timestamp_x", "timestamp_y"]:
        if c in df.columns:
            df = df.drop(columns=[c])

    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return out


if __name__ == "__main__":
    import sys
    try:
        out = fix()
        print("Wrote fixed CSV:", out)
    except Exception as e:
        print("ERROR:", e)
        sys.exit(1)
