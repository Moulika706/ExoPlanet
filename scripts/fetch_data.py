"""Fetch the Kepler KOI cumulative table from the NASA Exoplanet Archive.

Primary source: astroquery
(astroquery.ipac.nexsci.nasa_exoplanet_archive.NasaExoplanetArchive,
table="cumulative").
Fallback: the archive TAP service, SELECT * FROM cumulative returned as CSV.

Outputs (under data/raw/):
  koi_cumulative.csv     raw table
  koi_cumulative.csv.sha256  SHA-256 checksum of the CSV
  retrieval.json         retrieval date, source, row/column counts, checksum

The script also prints the schema, label counts and missing-value rates.
"""

from __future__ import annotations

import hashlib
import io
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from exovet.data import LABELS, TARGET, clean_labels

RAW_DIR = Path("data/raw")
RAW_CSV = RAW_DIR / "koi_cumulative.csv"
SHA_PATH = RAW_DIR / "koi_cumulative.csv.sha256"
RETRIEVAL_PATH = RAW_DIR / "retrieval.json"
TAP_URL = "https://exoplanetarchive.ipac.caltech.edu/tap/sync"


def fetch_via_astroquery() -> pd.DataFrame:
    """Query the KOI cumulative table via astroquery."""
    from astroquery.ipac.nexsci.nasa_exoplanet_archive import NasaExoplanetArchive

    table = NasaExoplanetArchive.query_criteria(table="cumulative", select="*")
    return table.to_pandas()


def fetch_via_tap() -> pd.DataFrame:
    """Fallback: download the cumulative table as CSV from the NEA TAP service."""
    params = {"query": "SELECT * FROM cumulative", "format": "csv"}
    response = requests.get(TAP_URL, params=params, timeout=180)
    response.raise_for_status()
    return pd.read_csv(io.BytesIO(response.content))


def fetch() -> tuple[pd.DataFrame, str]:
    """Fetch the table, trying astroquery first and TAP as fallback."""
    try:
        df = fetch_via_astroquery()
        source = (
            "astroquery.ipac.nexsci.nasa_exoplanet_archive."
            "query_criteria(table='cumulative', select='*')"
        )
        return df, source
    except Exception as exc:  # noqa: BLE001 - any failure triggers the TAP fallback
        print(f"astroquery fetch failed: {exc}; falling back to TAP CSV")
        df = fetch_via_tap()
        source = "https://exoplanetarchive.ipac.caltech.edu/tap/sync "
        "SELECT * FROM cumulative, format=csv"
        return df, source


def sha256_of_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def print_report(df: pd.DataFrame) -> None:
    print(f"rows: {len(df)}, columns: {len(df.columns)}")
    print("\nlabel counts:")
    print(df[TARGET].value_counts(dropna=False).to_string())
    print("\nmissing-value rates:")
    rates = df.isna().mean()
    missing = rates[rates > 0].sort_values(ascending=False)
    if missing.empty:
        print("  none")
    else:
        for column, rate in missing.items():
            print(f"  {column:<28} {rate:.4%}")
    print("\nschema:")
    for column in df.columns:
        print(f"  {column:<28} {str(df[column].dtype):<10}")
    kept = len(clean_labels(df))
    print(f"\nrows kept after cleaning to {list(LABELS)} labels: {kept}")


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    df, source = fetch()
    df.to_csv(RAW_CSV, index=False)

    digest = sha256_of_file(RAW_CSV)
    SHA_PATH.write_text(digest + "\n", encoding="utf-8")

    retrieval = {
        "retrieval_date": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "table": "cumulative",
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "sha256": digest,
    }
    RETRIEVAL_PATH.write_text(json.dumps(retrieval, indent=2) + "\n", encoding="utf-8")

    print_report(df)
    print(f"\nwrote {RAW_CSV}")
    print(f"wrote {SHA_PATH}")
    print(f"wrote {RETRIEVAL_PATH}")


if __name__ == "__main__":
    main()
