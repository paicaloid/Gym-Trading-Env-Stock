import gymnasium as gym
import numpy as np
import pandas as pd

from src.stock_trading_env.single_stock_env import SingleStockTradingEnv  # noqa

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

env = gym.make(
    "single-stock-v0",
    name="AAPL",
    df=df,
    windows=1,
    positions=[0, 0.5, 1],  # From -1 (=SHORT), to +1 (=LONG)
    initial_position=0,  # Initial position
    trading_fees=0,  # 0.01% per stock buy / sell
    # borrow_interest_rate=0.0003/100,  #per timestep (= 1h here)
    # reward_function=reward_function,
    portfolio_initial_value=100000,  # in FIAT (here, USD)
    # max_episode_duration=100,
    max_episode_steps=300,
)

env.add_metric(
    "Position Changes", lambda history: np.sum(np.diff(history["position"]) != 0)
)
env.add_metric("Episode Lenght", lambda history: len(history["position"]))
env.add_metric("Reward", lambda history: history["reward"][-1])
env.add_metric(
    "Portfolio Valuation", lambda history: history["portfolio_valuation"][-1]
)

episode = 1
for i in range(episode):
    print(f"episode : {i}")
    observation, info = env.reset()
    done, truncated = False, False

    while not done and not truncated:
        action = env.action_space.sample()
        observation, reward, done, truncated, info = env.step(action)
