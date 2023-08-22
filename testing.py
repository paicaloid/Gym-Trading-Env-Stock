import pandas as pd

# import numpy as np
from src.stock_trading_env.single_stock_env import SingleStockTradingEnv

df = pd.read_csv("examples/data/AAPL.csv", parse_dates=["date"], index_col="date")
df.sort_index(inplace=True)
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)

df_add = df.copy()
df_add["feature_close"] = df_add["close"].pct_change()
df_add["feature_open"] = df_add["open"] / df_add["close"]
df_add["feature_high"] = df_add["high"] / df_add["close"]
df_add["feature_low"] = df_add["low"] / df_add["close"]

env = SingleStockTradingEnv(
    df=df_add["2020-01-01":],
    positions=[0, 0.25, 0.5, 0.75, 1],
    portfolio_initial_value=1000000,
    initial_position=0,
)
print(env._nb_features)
print(env.observation_space)

observation, info = env.reset()
print(observation)
print(info)
# print(env._portfolio.cash, env._portfolio.size, env._portfolio.position)
