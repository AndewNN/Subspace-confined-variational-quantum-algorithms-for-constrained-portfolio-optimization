"""prepare_data.py — fetch the dataset used by experiments/PO_*.py from Yahoo Finance.

Produces two CSVs under ./dataset/ that the experiment scripts read as
"../dataset/top_50_us_stocks_data_20250526_011226_covariance.csv" and
"../dataset/top_50_us_stocks_returns_price.csv":

  Ticker, Average_Return, Price, Company_Name              (returns_price.csv)
  Ticker, AAPL, MSFT, ...   (covariance.csv  —  a square matrix)

The 50-ticker list below is a default placeholder of large-cap US equities; replace
TICKERS with the exact universe used in the paper to reproduce published numbers.

Run from the repo root:
    python prepare_data.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError:
    sys.exit("yfinance is required. Install with `pip install yfinance` "
             "or activate the conda env from environment.yml.")

# ---------------------------------------------------------------------------
# Replace the list below with the exact 50 tickers used in the paper.
# Defaults are large-cap US equities that span a price range comparable to the
# paper's [108, 216] USD band, sufficient for a smoke test of the pipeline.
# ---------------------------------------------------------------------------
TICKERS: list[str] = [
    'AAPL', 'ABBV', 'ABT', 'ACN', 'ADBE', 'AMD', 'AMZN', 'AVGO', 'BMY', 'BRK-B', 
    'CMCSA', 'COST', 'CRM', 'CSCO', 'CVX', 'DHR', 'DIS', 'GOOG', 'GOOGL', 'HD', 
    'INTC', 'JNJ', 'JPM', 'KO', 'LLY', 'MA', 'MCD', 'META', 'MRK', 'MSFT', 'NEE', 
    'NKE', 'NVDA', 'PEP', 'PFE', 'PG', 'PM', 'RTX', 'SPGI', 'T', 'TMO', 'TSLA', 
    'TXN', 'UNH', 'UPS', 'V', 'VZ', 'WFC', 'WMT', 'XOM',
]

START_DATE = "2015-04-01"
END_DATE = "2025-04-01"

COV_FILENAME = "top_50_us_stocks_data_20250526_011226_covariance.csv"
RET_FILENAME = "top_50_us_stocks_returns_price.csv"


def fetch_close_prices(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """Download adjusted-close daily prices from Yahoo Finance.

    Returns a DataFrame indexed by date with one column per ticker. Tickers
    with no data in the requested range are dropped (with a warning).
    """
    print(f"Downloading {len(tickers)} tickers from {start} to {end}...")
    df = yf.download(
        tickers, start=start, end=end,
        auto_adjust=True, progress=False, group_by="ticker", threads=True,
    )
    if isinstance(df.columns, pd.MultiIndex):
        # group_by="ticker" produces a MultiIndex (ticker, field). Pull Close.
        closes = pd.concat({t: df[t]["Close"] for t in tickers if t in df.columns.levels[0]}, axis=1)
    else:
        # Single-ticker download falls back to a flat DataFrame.
        closes = df[["Close"]].rename(columns={"Close": tickers[0]})
    dropped = [t for t in tickers if t not in closes.columns]
    if dropped:
        print(f"  warning: no data for {dropped} (dropped from universe)")
    return closes.dropna(how="all")


def build_returns_price(closes: pd.DataFrame) -> pd.DataFrame:
    """Build the per-ticker summary table the experiments expect."""
    daily_returns = closes.pct_change().dropna(how="all")
    last_price = closes.ffill().iloc[-1]
    avg_return = daily_returns.mean()
    out = pd.DataFrame({
        "Ticker": closes.columns,
        "Average_Return": avg_return.values,
        "Price": last_price.values,
        # Company_Name is informational only; keep blank to avoid hammering
        # yfinance with one .info call per ticker.
        "Company_Name": closes.columns,
    })
    return out


def build_covariance(closes: pd.DataFrame) -> pd.DataFrame:
    """Build the daily-return covariance matrix in the wide format used by the experiments."""
    daily_returns = closes.pct_change().dropna(how="all")
    cov = daily_returns.cov()
    cov.insert(0, "Ticker", cov.index)
    cov = cov.reset_index(drop=True)
    return cov


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-dir", default="dataset",
                        help="Directory to write CSV files into (default: dataset)")
    parser.add_argument("--start", default=START_DATE)
    parser.add_argument("--end", default=END_DATE)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    closes = fetch_close_prices(TICKERS, args.start, args.end)

    ret_df = build_returns_price(closes)
    ret_path = out_dir / RET_FILENAME
    ret_df.to_csv(ret_path, index=False)
    print(f"Wrote {ret_path}  ({len(ret_df)} rows)")

    cov_df = build_covariance(closes)
    cov_path = out_dir / COV_FILENAME
    cov_df.to_csv(cov_path, index=False)
    print(f"Wrote {cov_path}  ({len(cov_df)} rows × {len(cov_df.columns)} cols)")

    print("\nDone. The experiment scripts can now find the CSVs at ../dataset/...")


if __name__ == "__main__":
    main()
