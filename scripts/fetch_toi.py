"""Fetch the TESS Objects of Interest (TOI) table from the NASA Exoplanet Archive.

Primary source: astroquery
(astroquery.ipac.nexsci.nasa_exoplanet_archive.NasaExoplanetArchive,
table="toi").
Fallback: the archive TAP service, SELECT * FROM toi returned as CSV.

Outputs (under data/raw/):
  toi.csv             raw table
  toi.csv.sha256      SHA-256 checksum of the CSV
  retrieval_toi.json  retrieval date, source, row/column counts, checksum

The TOI table is the cross-mission test set of the project: the model is
trained on Kepler and applied to these objects without retraining.
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

from exovet.crossmission import DISPOSITION_TO_LABEL

RAW_DIR = Path("data/raw")
RAW_CSV = RAW_DIR / "toi.csv"
SHA_PATH = RAW_DIR / "toi.csv.sha256"
RETRIEVAL_PATH = RAW_DIR / "retrieval_toi.json"
# The path is case sensitive: /TAP/sync answers, /tap/sync is a 404.
TAP_URL = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"


def fetch_via_astroquery() -> pd.DataFrame:
    """Query the TOI table via astroquery."""
    from astroquery.ipac.nexsci.nasa_exoplanet_archive import NasaExoplanetArchive

    table = NasaExoplanetArchive.query_criteria(table="toi", select="*")
    return table.to_pandas()


def fetch_via_tap() -> pd.DataFrame:
    """Fallback: download the TOI table as CSV from the NEA TAP service."""
    params = {"query": "SELECT * FROM toi", "format": "csv"}
    response = requests.get(TAP_URL, params=params, timeout=600)
    response.raise_for_status()
    return pd.read_csv(io.BytesIO(response.content))


def fetch() -> tuple[pd.DataFrame, str]:
    """Fetch the table, trying astroquery first and TAP as fallback."""
    try:
        df = fetch_via_astroquery()
        source = (
            "astroquery.ipac.nexsci.nasa_exoplanet_archive.query_criteria(table='toi', select='*')"
        )
        return df, source
    except Exception as exc:  # noqa: BLE001 - any failure triggers the TAP fallback
        print(f"astroquery fetch failed: {exc}; falling back to TAP CSV")
        df = fetch_via_tap()
        source = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync SELECT * FROM toi, format=csv"
        return df, source


def sha256_of_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def print_report(df: pd.DataFrame) -> None:
    print(f"rows: {len(df)}, columns: {len(df.columns)}")
    if "tfopwg_disp" in df.columns:
        print("\nTFOPWG disposition counts:")
        print(df["tfopwg_disp"].value_counts(dropna=False).to_string())
        print("\nmapped to the canonical labels:")
        mapped = df["tfopwg_disp"].map(DISPOSITION_TO_LABEL).dropna()
        print(mapped.value_counts().to_string())
    print("\nmissing-value rates of the shared columns:")
    from exovet.crossmission import TOI_COLUMN_FOR_FEATURE

    shared = [c for c in TOI_COLUMN_FOR_FEATURE.values() if c in df.columns]
    rates = df[shared].isna().mean().sort_values(ascending=False)
    for column, rate in rates.items():
        print(f"  {column:<16} {rate:.4%}")


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    df, source = fetch()
    df.to_csv(RAW_CSV, index=False)

    digest = sha256_of_file(RAW_CSV)
    SHA_PATH.write_text(digest + "\n", encoding="utf-8")

    retrieval = {
        "retrieval_date": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "table": "toi",
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "sha256": digest,
        "disposition_counts": {
            str(key): int(value)
            for key, value in df.get("tfopwg_disp", pd.Series(dtype=object))
            .value_counts(dropna=False)
            .items()
        },
    }
    RETRIEVAL_PATH.write_text(json.dumps(retrieval, indent=2) + "\n", encoding="utf-8")

    print_report(df)
    print(f"\nwrote {RAW_CSV}")
    print(f"wrote {SHA_PATH}")
    print(f"wrote {RETRIEVAL_PATH}")


if __name__ == "__main__":
    main()
