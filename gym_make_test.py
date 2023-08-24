import gymnasium as gym
import pandas as pd

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

env = gym.make(
    "single-stock-v0",
    df=df_add["2020-01-01":],
    positions=[0, 0.25, 0.5, 0.75, 1],
    portfolio_initial_value=1000000,
)

done, truncated = False, False
observation, info = env.reset()
count = 0
while not done and not truncated:
    count += 1
    action = env.action_space.sample()
    observation, reward, done, truncated, info = env.step(action)
    print(info)

    if count == 10:
        break
