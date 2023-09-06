# import datetime
# from datetime import datetime
# from typing import Callable, Dict, List, Optional, Tuple, Type, Union

# import gym
# import gymnasium as gym
# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# import pandas_ta as ta
# import tensorflow as tf
# import torch as th
# from sklearn.preprocessing import MinMaxScaler
# from stable_baselines3 import PPO
# from stable_baselines3.common.callbacks import EvalCallback
# from stable_baselines3.common.evaluation import evaluate_policy
# from stable_baselines3.common.monitor import Monitor
# from stable_baselines3.common.policies import ActorCriticPolicy
# from torch import nn

# import gym_trading_env
# from agent import Agent
# from gym_trading_env.downloader import download
# from gym_trading_env.renderer import Renderer
# from src.stock_trading_env.single_stock_env import SingleStockTradingEnv

from training_rl import env_create, eval_callback_create, test_process, train_process
from src.utils.preprocess import preprocess_dataframe, train_test_dataframe

# def process_df(filepath):
#     df = pd.read_csv(filepath, parse_dates=["date"], index_col="date")
#     df.sort_index(inplace=True)
#     df.dropna(inplace=True)
#     df.drop_duplicates(inplace=True)
#     df["feature_close"] = df["close"].pct_change()
#     df["feature_open"] = df["open"] / df["close"]
#     df["feature_high"] = df["high"] / df["close"]
#     df["feature_low"] = df["low"] / df["close"]
#     rsi_values = ta.rsi(df["close"]).values.reshape(-1, 1)
#     scaler = MinMaxScaler()
#     df["feature_rsi"] = scaler.fit_transform(rsi_values)
#     df.dropna(inplace=True)
#     return df


# df = process_df("examples/data/AAPL.csv")

df = preprocess_dataframe("example/data/AAPL.csv", indicators=True)
train_df, test_df = train_test_dataframe(df, train_test_split=0.8)
# ------train-----#
# train_df = df["2015-01-01":"2020-01-01"]
# env_train = env_create(
#     "single-stock-v0",
#     name="AAPL_1",
#     df=train_df,
#     windows=1,
#     positions=[0, 1],
#     initial_position=0,
#     trading_fees=0,
#     portfolio_initial_value=100000,
#     max_episode_duration="max",
#     verbose=1,
# )

# policy_kwargs = dict(activation_fn=th.nn.ReLU, net_arch=[256, 128])
eval_callback = EvalCallback(
    env_train,
    best_model_save_path="./models/",
    log_path="./logs/",
    eval_freq=500,
    deterministic=False,
    render=False,
)
# train_process(env_train, policy_kwargs, eval_callback)


# ------test-----#
# test_df = df["2021-01-01":]
# env_test = env_create(
#     "single-stock-v0",
#     name="AAPL_1",
#     df=test_df,
#     windows=1,
#     positions=[0, 1],  # From -1 (=SHORT), to +1 (=LONG)
#     initial_position=0,  # Initial position
#     trading_fees=0,  # 0.01% per stock buy / sell
#     # borrow_interest_rate=0.0003/100,  #per timestep (= 1h here)
#     # reward_function=reward_function,
#     portfolio_initial_value=100000,  # in FIAT (here, USD)
#     max_episode_duration="max",
#     verbose=1,
# )

# env_test.add_metric("Reward", lambda history: history["reward"][-1])
# env_test.add_metric(
#     "Portfolio Valuation", lambda history: history["portfolio_valuation"][-1]
# )


# policy_kwargs = dict(activation_fn=th.nn.ReLU, net_arch=[256, 256])

# model = PPO.load("models/best_model.zip")
# env = Monitor(env_test)
# for rep in range(5):
#     print(f"\nRepetition {rep}")

#     model = model

#     # evaluate
#     mean_reward, std_reward = evaluate_policy(
#         model, env, n_eval_episodes=10, deterministic=False
#     )
#     print(f"Eval reward: {mean_reward} (+/-{std_reward})")

# reward_list = []
# for i in range(10):
    env, reward = test_process(env, model)
#     print(f"reward: {reward}")
#     reward_list.append(reward)


# print(reward_list)

#     env.save_for_render()

# renderer = Renderer(render_logs_dir="render_logs")
# renderer.run()
