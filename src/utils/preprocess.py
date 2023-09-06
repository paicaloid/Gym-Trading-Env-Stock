from typing import Optional

import pandas as pd


def example_dataframe() -> pd.DataFrame:
    df = pd.read_csv("examples/data/AAPL.csv", parse_dates=["date"], index_col="date")
    df.sort_index(inplace=True)
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    df["feature_close"] = df["close"].pct_change()
    df["feature_open"] = df["open"] / df["close"]
    df["feature_high"] = df["high"] / df["close"]
    df["feature_low"] = df["low"] / df["close"]
    df = df["2020-01-01":]
    df.dropna(inplace=True)
    return df


def preprocess_dataframe(
    path: str,
    indicators: bool = False,
) -> pd.DataFrame:
    # [ ] Add check for file with pathlib
    # [ ] check if file is csv and verify columns

    df = pd.read_csv(path, parse_dates=["date"], index_col="date")
    df.sort_index(inplace=True)
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    df["feature_close"] = df["close"].pct_change()
    df["feature_open"] = df["open"] / df["close"]
    df["feature_high"] = df["high"] / df["close"]
    df["feature_low"] = df["low"] / df["close"]

    if indicators:
        # [ ] Add indicators
        pass

    df.dropna(inplace=True)
    return df


def train_test_dataframe(
    df: pd.DataFrame,
    train_test_split: Optional[float] = None,
    train_start: str = None,
    test_start: str = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pass
