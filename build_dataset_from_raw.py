"""Rebuild dataset/ from the raw daily-bar file.

This is the derivation that actually produced the two CSVs shipped in
``dataset/``. It takes the raw long-format daily file

    Date, Ticker, Close, Daily_Return, Company_Name

restricts it to the paper's window (2015-04-01 to 2025-04-01), and writes

    dataset/top_50_us_stocks_returns_price.csv
        Ticker, Average_Return, Price, Company_Name
        Average_Return = mean Daily_Return per ticker over the window
        Price          = last Close per ticker in the window

    dataset/top_50_us_stocks_data_20250526_011226_covariance.csv
        the 50 x 50 covariance of Daily_Return, tickers in rows and columns

Two uses:

1.  ``--verify`` checks a candidate raw file against the shipped CSVs. If the
    regenerated values match, that raw file is the one the paper was built
    from. This is the way to confirm a recovered archive is the right one.

2.  Without ``--verify`` it writes the CSVs, for extending the study to a
    different window or universe.

Note that ``prepare_data.py`` is a different route to the same outputs: it
downloads from Yahoo Finance and derives the columns itself. That path cannot
reproduce the shipped files, because adjusted-price histories are revised over
time. This script is the faithful derivation; use it when you have the raw
file.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

START_DATE = "2015-04-01"
END_DATE = "2025-04-01"
RET_FILENAME = "top_50_us_stocks_returns_price.csv"
COV_FILENAME = "top_50_us_stocks_data_20250526_011226_covariance.csv"
REQUIRED = ["Date", "Ticker", "Close", "Daily_Return", "Company_Name"]


def build(raw: pd.DataFrame, start: str, end: str):
    """Exactly the derivation of process_csv.ipynb."""
    missing = [c for c in REQUIRED if c not in raw.columns]
    if missing:
        sys.exit(f"raw file is missing required column(s): {missing}\n"
                 f"expected: {REQUIRED}\nfound:    {list(raw.columns)}")

    data = raw[(raw["Date"] >= start) & (raw["Date"] <= end)]
    if data.empty:
        sys.exit(f"no rows in the window {start} .. {end}")

    grouped = data.groupby("Ticker")
    returns_price = grouped.agg({"Daily_Return": "mean"}).reset_index()
    returns_price.columns = ["Ticker", "Average_Return"]
    returns_price["Price"] = grouped["Close"].last().values
    returns_price["Company_Name"] = grouped["Company_Name"].first().values

    covariance = (data.pivot(index="Date", columns="Ticker", values="Daily_Return")
                      .cov()
                      .reset_index())
    return returns_price, covariance


def compare(new: pd.DataFrame, ref_path: Path, key: str, tol: float) -> bool:
    """Numeric comparison against a shipped CSV; returns True if they agree."""
    if not ref_path.exists():
        print(f"  [skip] {ref_path} not present")
        return True
    ref = pd.read_csv(ref_path)
    if list(new.columns) != list(ref.columns):
        print(f"  [FAIL] {ref_path.name}: column mismatch\n"
              f"         regenerated {list(new.columns)}\n         shipped     {list(ref.columns)}")
        return False
    if len(new) != len(ref):
        print(f"  [FAIL] {ref_path.name}: {len(new)} rows regenerated vs {len(ref)} shipped")
        return False
    a = new.sort_values(key).reset_index(drop=True)
    b = ref.sort_values(key).reset_index(drop=True)
    if not a[key].equals(b[key]):
        print(f"  [FAIL] {ref_path.name}: '{key}' values differ")
        return False
    num = [c for c in a.columns if pd.api.types.is_numeric_dtype(a[c])]
    diff = np.abs(a[num].to_numpy(float) - b[num].to_numpy(float))
    worst = float(np.nanmax(diff)) if diff.size else 0.0
    ok = worst <= tol
    print(f"  [{'OK  ' if ok else 'FAIL'}] {ref_path.name}: largest absolute difference {worst:.3e} "
          f"(tolerance {tol:g})")
    return ok


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("raw_csv", help="the raw *_with_returns.csv daily-bar file")
    ap.add_argument("--out-dir", default="dataset", help="where to write (default: dataset)")
    ap.add_argument("--start", default=START_DATE)
    ap.add_argument("--end", default=END_DATE)
    ap.add_argument("--verify", action="store_true",
                    help="compare against the shipped CSVs instead of writing")
    ap.add_argument("--tol", type=float, default=1e-10,
                    help="absolute tolerance used by --verify (default: 1e-10)")
    args = ap.parse_args()

    raw_path = Path(args.raw_csv)
    if not raw_path.exists():
        sys.exit(f"no such file: {raw_path}")
    raw = pd.read_csv(raw_path)
    print(f"read {raw_path}  ({len(raw):,} rows, {raw['Ticker'].nunique()} tickers)"
          if "Ticker" in raw.columns else f"read {raw_path}  ({len(raw):,} rows)")

    returns_price, covariance = build(raw, args.start, args.end)
    print(f"derived {len(returns_price)} tickers over {args.start} .. {args.end}")

    out = Path(args.out_dir)
    if args.verify:
        print(f"\nverifying against {out}/ :")
        ok = compare(returns_price, out / RET_FILENAME, "Ticker", args.tol)
        ok &= compare(covariance, out / COV_FILENAME, "Ticker", args.tol)
        print("\n" + ("MATCH: this raw file reproduces the shipped dataset."
                      if ok else
                      "MISMATCH: this raw file does NOT reproduce the shipped dataset."))
        sys.exit(0 if ok else 1)

    out.mkdir(parents=True, exist_ok=True)
    returns_price.to_csv(out / RET_FILENAME, index=False)
    covariance.to_csv(out / COV_FILENAME, index=False)
    print(f"wrote {out/RET_FILENAME}\nwrote {out/COV_FILENAME}")


if __name__ == "__main__":
    main()
