import gzip
import shutil
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import requests


def fetch_openaq_location_ml(location_id: int, start_date: str, output_dir: str | Path):
    """Download archived OpenAQ location records and produce an hourly ML CSV.

    This is a thin, reusable refactor of the previous top-level script so it can be
    invoked programmatically from the project's central downloader.
    """
    BASE_URL = "https://openaq-data-archive.s3.amazonaws.com"
    OUTPUT_DIR = Path(output_dir)
    RAW_CSV = OUTPUT_DIR / "raw_long_format.csv"
    ML_CSV = OUTPUT_DIR / "ml_wide_format.csv"

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not RAW_CSV.exists():
        pd.DataFrame(columns=["datetime", "parameter", "value"]).to_csv(RAW_CSV, index=False)

    def fetch_day(date_obj: datetime):
        yyyymmdd = date_obj.strftime("%Y%m%d")
        year = date_obj.strftime("%Y")
        month = date_obj.strftime("%m")

        filename = f"location-{location_id}-{yyyymmdd}.csv.gz"

        url = (
            f"{BASE_URL}/records/csv.gz/"
            f"locationid={location_id}/"
            f"year={year}/"
            f"month={month}/"
            f"{filename}"
        )

        try:
            r = requests.get(url, timeout=60)

            if r.status_code != 200:
                return None

            gz_path = OUTPUT_DIR / filename
            csv_path = gz_path.with_suffix("")

            with open(gz_path, "wb") as f:
                f.write(r.content)

            with gzip.open(gz_path, "rb") as f_in:
                with open(csv_path, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)

            df = pd.read_csv(csv_path)

            try:
                gz_path.unlink()
            except Exception:
                pass
            try:
                csv_path.unlink()
            except Exception:
                pass

            if df.empty:
                return None

            return df

        except Exception:
            return None

    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    today = datetime.today()

    current = start_dt

    while current <= today:
        df = fetch_day(current)
        if df is not None:
            if "datetime" in df.columns:
                df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
            elif "date.utc" in df.columns:
                df["datetime"] = pd.to_datetime(df["date.utc"], utc=True, errors="coerce")
            else:
                current += timedelta(days=1)
                continue

            df = df.dropna(subset=["datetime"]) 

            param_col = None
            value_col = None
            for c in df.columns:
                if c.lower() in ["parameter", "param"]:
                    param_col = c
                if c.lower() == "value":
                    value_col = c

            if param_col and value_col:
                tmp = df[["datetime", param_col, value_col]].copy()
                tmp.columns = ["datetime", "parameter", "value"]
                tmp.to_csv(RAW_CSV, mode="a", header=False, index=False)

        current += timedelta(days=1)

    # Load and pivot to ML wide format
    raw_df = pd.read_csv(RAW_CSV)
    raw_df["datetime"] = pd.to_datetime(raw_df["datetime"], utc=True, errors="coerce")
    raw_df = raw_df.dropna(subset=["datetime", "parameter", "value"]) 
    raw_df["value"] = pd.to_numeric(raw_df["value"], errors="coerce")
    raw_df = raw_df.dropna(subset=["value"]) 

    agg = raw_df.groupby(["datetime", "parameter"], as_index=False).mean()
    ml_df = agg.pivot(index="datetime", columns="parameter", values="value")
    ml_df = ml_df.sort_index()
    ml_df = ml_df.resample("1H").mean()
    ml_df = ml_df.interpolate(method="time")
    ml_df = ml_df.ffill().bfill()
    ml_df.to_csv(ML_CSV)

    return ML_CSV
