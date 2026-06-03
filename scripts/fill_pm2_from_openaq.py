"""Fill the `pm2` column in the merged historical CSV using `pm2_openaq` values."""
from pathlib import Path
import pandas as pd


def fill(input_path: str | Path = "data/raw/historical_rawdata_with_openaq_fixed.csv", output_path: str | Path = "data/raw/historical_rawdata_pm2_filled.csv"):
    inp = Path(input_path)
    out = Path(output_path)
    df = pd.read_csv(inp, low_memory=False)
    if "pm2_openaq" not in df.columns:
        raise RuntimeError("pm2_openaq column missing")
    if "pm2" not in df.columns:
        df["pm2"] = pd.NA
    df["pm2"] = df["pm2"].fillna(df["pm2_openaq"])
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return out


if __name__ == "__main__":
    import sys
    try:
        out = fill()
        print("Wrote filled CSV:", out)
    except Exception as e:
        print("ERROR:", e)
        sys.exit(1)
