from datetime import datetime
import calendar
from io import StringIO
from pathlib import Path
import pandas as pd


def scrape_environment_canada(climate_id: str, province: str, start_year: int, end_year: int | None = None, output_file: str | Path = "canada_climate_hourly.csv"):
    """Scrape hourly climate data from Environment Canada for the specified station.

    Uses the same logic as the previous SWOB_api.py but wrapped in a callable function.
    Note: this function uses Selenium and webdriver-manager; to avoid heavy imports on
    module load the Selenium imports are performed inside the function.
    """
    if end_year is None:
        end_year = datetime.now().year

    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.chrome.service import Service
    from webdriver_manager.chrome import ChromeDriverManager
    import time

    BASE_URL = "https://climate.weather.gc.ca/climate_data/hourly_data_e.html"

    chrome_options = Options()
    chrome_options.add_argument("--headless=new")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--window-size=1920,1080")

    driver = webdriver.Chrome(
        service=Service(ChromeDriverManager().install()),
        options=chrome_options,
    )

    all_data = []

    current_year = datetime.now().year
    current_month = datetime.now().month
    current_day = datetime.now().day

    for year in range(start_year, end_year + 1):
        max_month = current_month if year == current_year else 12
        for month in range(1, max_month + 1):
            days_in_month = calendar.monthrange(year, month)[1]
            max_day = days_in_month
            if year == current_year and month == current_month:
                max_day = min(current_day, days_in_month)
            for day in range(1, max_day + 1):
                url = (
                    f"{BASE_URL}?timeframe=1"
                    f"&Prov={province}"
                    f"&climate_id={climate_id}"
                    f"&Year={year}"
                    f"&Month={month}"
                    f"&Day={day}"
                )
                try:
                    driver.get(url)
                    time.sleep(4)
                    table = driver.find_element(By.CLASS_NAME, "table")
                    table_html = table.get_attribute("outerHTML")
                    df = pd.read_html(StringIO(table_html))[0]
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = [
                            "_".join([str(x) for x in col if str(x) != "nan"]).strip()
                            for col in df.columns.values
                        ]
                    df.columns = [str(col).strip() for col in df.columns]
                    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
                    df["year"] = year
                    df["month"] = month
                    df["day"] = day

                    time_col = None
                    for col in df.columns:
                        lower_col = col.lower()
                        if "time" in lower_col and "lst" in lower_col:
                            time_col = col
                            break

                    if time_col:
                        timestamp_text = (
                            df["year"].astype(str)
                            + "-"
                            + df["month"].astype(str).str.zfill(2)
                            + "-"
                            + df["day"].astype(str).str.zfill(2)
                            + " "
                            + df[time_col].astype(str)
                        )
                        df["datetime"] = pd.to_datetime(timestamp_text, errors="coerce")
                    else:
                        df["datetime"] = pd.NaT

                    df = df.dropna(how="all")
                    all_data.append(df)

                except Exception:
                    continue

    driver.quit()

    if len(all_data) == 0:
        return None

    final_df = pd.concat(all_data, ignore_index=True)
    final_df = final_df.drop_duplicates()
    if "datetime" in final_df.columns:
        final_df = final_df.sort_values("datetime")
    final_df = final_df.reset_index(drop=True)
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(output_file, index=False)
    return Path(output_file)
