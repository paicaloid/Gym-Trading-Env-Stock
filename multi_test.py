import gymnasium as gym
import pandas as pd
import numpy as np

from src.stock_trading_env.multi_stock_env import MultiStockTradingEnv


df_AAPL = pd.read_csv("examples/data/AAPL.csv", parse_dates=["date"], index_col="date")
df_AAPL.sort_index(inplace=True)
df_AAPL.dropna(inplace=True)
df_AAPL.drop_duplicates(inplace=True)
df_AAPL["feature_close"] = df_AAPL["close"].pct_change()
df_AAPL["feature_open"] = df_AAPL["open"] / df_AAPL["close"]
df_AAPL["feature_high"] = df_AAPL["high"] / df_AAPL["close"]
df_AAPL["feature_low"] = df_AAPL["low"] / df_AAPL["close"]

df_TSLA = pd.read_csv("examples/data/TSLA.csv", parse_dates=["date"], index_col="date")
df_TSLA.sort_index(inplace=True)
df_TSLA.dropna(inplace=True)
df_TSLA.drop_duplicates(inplace=True)
df_TSLA["feature_close"] = df_TSLA["close"].pct_change()
df_TSLA["feature_open"] = df_TSLA["open"] / df_TSLA["close"]
df_TSLA["feature_high"] = df_TSLA["high"] / df_TSLA["close"]
df_TSLA["feature_low"] = df_TSLA["low"] / df_TSLA["close"]

df = pd.concat([df_AAPL, df_TSLA], axis=1, keys=["AAPL", "TSLA"])

env = MultiStockTradingEnv(
    df=df["2020-01-01":],
    positions=[0, 0.5, 1],
    trading_fees=0.000,
    portfolio_initial_value=1000000,
    max_episode_duration=100,
)

ob, info = env.reset()

# print(ob)
print(info)

ob, reward, done, truncated, info = env.step(np.array([2, 0]))
# print(ob)
print(info)

ob, reward, done, truncated, info = env.step(np.array([0, 1]))
# print(ob)
print(info)
