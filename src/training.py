import time

import gymnasium as gym
import numpy as np
import pandas as pd

from ppo_agent.agent import Agent
from stock_trading_env.single_stock_env import SingleStockTradingEnv  # noqa


def reward_function(history):
    return np.log(
        history["portfolio_valuation", -1] / history["portfolio_valuation", -2]
    )


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
    "single-stock-v1",
    name="AAPL",
    df=df,
    windows=1,
    positions=[0, 1],
    initial_position="random",
    trading_fees=0,
    portfolio_initial_value=100000,
)

env.add_metric(
    "Position Changes", lambda history: np.sum(np.diff(history["position"]) != 0)
)
env.add_metric("Episode Lenght", lambda history: len(history["position"]))
env.add_metric("Reward", lambda history: history["reward"][-1])
env.add_metric(
    "Portfolio Valuation", lambda history: history["portfolio_valuation"][-1]
)

N = 20
batch_size = 64
n_epochs = 4
alpha = 0.0003
score_history = []
learn_iters = 0
avg_score = 0
n_steps = 0

agent = Agent(
    n_actions=env.action_space.n,
    batch_size=batch_size,
    alpha=alpha,
    n_epochs=n_epochs,
    input_dims=env.observation_space.shape,
)

best_score = env.reward_range[0]
score = 0
episode = 5
for i in range(episode):
    st = time.process_time()
    print(f"episode : {i}")
    observation, info = env.reset()
    done = False
    truncated = False
    while not done and not truncated:
        action, prob, val = agent.choose_action(observation)

        observation_, reward, done, truncated, info = env.step(action)

        n_steps += 1
        score += reward
        agent.remember(observation, action, prob, val, reward, done)

        if n_steps % N == 0:
            agent.learn()
            learn_iters += 1
        observation = observation_

    score_history.append(score)
    print(f"episode time : {time.process_time() - st}")
