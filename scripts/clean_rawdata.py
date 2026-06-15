"""Clean the historical RawData.csv into the project schema."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from infrastructure.kafka.scripts.data_downloader import DEFAULT_KEEP_COLUMNS, clean_raw_data


def main() -> None:
    source = ROOT / "RawData.csv"
    target = ROOT / "data" / "raw" / "historical_rawdata.csv"
    clean_raw_data(source, target, DEFAULT_KEEP_COLUMNS)
    print(f"Cleaned dataset written to {target}")


if __name__ == "__main__":
    main()
