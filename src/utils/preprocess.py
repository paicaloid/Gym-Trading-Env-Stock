import os
from typing import Optional

import pandas as pd
import pandas_ta as ta
from sklearn.preprocessing import MinMaxScaler


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
    # [x] Add check for file with pathlib
    # [x] check if file is csv and verify columns
    if not os.path.exists(path):
        raise FileNotFoundError(f"File {path} not found")

    columns = ["date", "open", "high", "low", "close", "volume"]
    if columns != pd.read_csv(path).columns.tolist():
        raise ValueError(f"Columns of {path} are not correct. Should be {columns}")

    df = pd.read_csv(path, parse_dates=["date"], index_col="date")
    df.sort_index(inplace=True)
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    df["feature_close"] = df["close"].pct_change()
    df["feature_open"] = df["open"] / df["close"]
    df["feature_high"] = df["high"] / df["close"]
    df["feature_low"] = df["low"] / df["close"]

    if indicators:
        # [x] Add indicators
        rsi_values = ta.rsi(df["close"]).values.reshape(-1, 1)
        scaler = MinMaxScaler()
        df["feature_rsi"] = scaler.fit_transform(rsi_values)

    df.dropna(inplace=True)
    return df


def train_test_dataframe(
    df: pd.DataFrame,
    train_test_split: Optional[float] = None,
    train_start: str = None,
    test_start: str = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if train_test_split is not None:
        train_df = df[: int(len(df) * train_test_split)]
        test_df = df[int(len(df) * train_test_split) :]

    else:
        train_df = df[train_start:test_start]
        test_df = df[test_start:]

    return train_df, test_df
