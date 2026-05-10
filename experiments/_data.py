"""Dataset loaders shared by the experiment scripts.

The two CSVs produced by ``prepare_data.py`` (returns_price + covariance)
are read identically by several PO_*.py scripts. Centralising the load +
filter + alignment logic here means a single point to update if the
dataset format ever changes.

All paths are relative to ``experiments/`` (the standard cwd convention).
"""
from __future__ import annotations

from typing import Tuple

import pandas as pd

DATASET_DIR = "../dataset"
RETURNS_PRICE_CSV = f"{DATASET_DIR}/top_50_us_stocks_returns_price.csv"
COVARIANCE_CSV = f"{DATASET_DIR}/top_50_us_stocks_data_20250526_011226_covariance.csv"


def load_universe(min_price: float, max_price: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load the asset universe and filter to a price band.

    Reads both dataset CSVs, keeps only tickers with last close price in
    ``(min_price, max_price)``, and reindexes the covariance matrix to
    match the filtered ticker set in row and column order.

    Returns
    -------
    returns_price : DataFrame
        Filtered ``returns_price`` table — columns ``Ticker, Average_Return,
        Price, Company_Name``.
    covariance : DataFrame
        Square covariance matrix (rows × cols both = filtered tickers) with
        the leading ``Ticker`` column preserved.
    """
    returns_price = pd.read_csv(RETURNS_PRICE_CSV)
    covariance = pd.read_csv(COVARIANCE_CSV)

    returns_price = returns_price[
        (returns_price["Price"] > min_price) & (returns_price["Price"] < max_price)
    ].reset_index(drop=True)

    covariance = covariance.loc[
        covariance["Ticker"].isin(returns_price["Ticker"])
    ].reset_index(drop=True)
    covariance = covariance[["Ticker"] + covariance["Ticker"].tolist()]

    return returns_price, covariance
