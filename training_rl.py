import datetime
from datetime import datetime
from typing import Callable, Dict, List, Optional, Tuple, Type, Union

import gym
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas_ta as ta
import tensorflow as tf
import torch as th
from sklearn.preprocessing import MinMaxScaler
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.policies import ActorCriticPolicy
from torch import nn

import gym_trading_env
from agent import Agent
from gym_trading_env.downloader import download
from gym_trading_env.renderer import Renderer
from src.stock_trading_env.single_stock_env import SingleStockTradingEnv


def env_create(
    env_name,
    name,
    df,
    windows,
    positions,
    initial_position,
    trading_fees,
    portfolio_initial_value,
    max_episode_duration,
    verbose,
):
    env = gym.make(
        env_name,
        name=name,
        df=df,
        windows=windows,
        positions=positions,  # From -1 (=SHORT), to +1 (=LONG)
        initial_position=initial_position,  # Initial position
        trading_fees=trading_fees,  # 0.01% per stock buy / sell
        # borrow_interest_rate=0.0003/100,  #per timestep (= 1h here)
        # reward_function=reward_function,
        portfolio_initial_value=portfolio_initial_value,  # in FIAT (here, USD)
        max_episode_duration="max",
        verbose=1,
    )
    return env


def train_process(env, policy_kwargs, eval_callback):
    env = Monitor(env)
    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        verbose=0,
        n_epochs=5,
        tensorboard_log="./PPO_tensorboard/",
    )

    return model.learn(total_timesteps=100000, callback=eval_callback)


def eval_callback_create(env_):
    return EvalCallback(
        env=env_,
        best_model_save_path="./models/",
        log_path="./logs/",
        eval_freq=100,
        deterministic=False,
        render=False,
    )


def test_process(env, model):
    reward_tot = 0
    obs, info = env.reset()
    done = False
    truncated = False
    while not done and not truncated:
        obs = obs[np.newaxis, ...]
        action, _ = model.predict(obs)
        obs, reward, done, truncated, info = env.step(action[0])
        reward_tot += reward

    return env, reward_tot
