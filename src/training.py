# import time

import gymnasium as gym
import numpy as np
import pandas as pd
from stable_baselines3 import PPO

# from ppo_agent.agent import Agent  # noqa
from stock_trading_env.single_stock_env import SingleStockTradingEnv  # noqa
from utils.enviroments import env_create

# from utils import example_dataframe
from utils.preprocess import preprocess_dataframe

df = preprocess_dataframe(
    path="examples/data/AAPL.csv",
    indicators=False,
)

print(df)

env_train = env_create(
    "single-stock-v1",
    name="AAPL_1",
    df=df,
    windows=1,
    positions=[0, 1],
    initial_position=0,
    trading_fees=0,
    portfolio_initial_value=100000,
    max_episode_duration="max",
    verbose=1,
)
