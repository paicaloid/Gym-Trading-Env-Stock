# *** dont remember add rsi normalize
import pandas as pd
from stable_baselines3 import A2C, PPO
from src.stock_trading_env.single_stock_env import SingleStockTradingEnv  # noqa: F401, E501
from gym_trading_env.environments import TradingEnv  # noqa: F401
import gymnasium as gym
import os
import time
import numpy as np


def _read_data(file="ADVANC.csv"):
    df = pd.read_csv(file, parse_dates=['datetime'], index_col="datetime")
    df.sort_index(inplace=True)
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    return df


def _add_feature(df):
    df_add = df.copy()
    # df_add["feature_close"] = df_add["close"].pct_change()
    df_add["feature_open"] = df_add["open"] / df_add["close"]
    df_add["feature_high"] = df_add["high"] / df_add["close"]
    df_add["feature_low"] = df_add["low"] / df_add["close"]
    return df_add


def _train_model(df_add, policy="A2C", id="TradingEnv"):
    models_dir = f"models/{id}-{policy}-{int(time.time())}"
    logdir = f"logs/{id}-{policy}-{int(time.time())}"

    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    if not os.path.exists(logdir):
        os.makedirs(logdir)

    env = gym.make(
        id,
        name="ADVANC",
        df=df_add["2020-01-01":],
        positions=[0, 0.25, 0.5, 0.75, 1],
        # positions=[0, 1],
        portfolio_initial_value=1000000,
        initial_position=0,
    )
    TIMESTEPS = 1000
    env.reset()
    if policy == "A2C":
        model = A2C("MlpPolicy", env, verbose=1, tensorboard_log=logdir)
    elif policy == "PPO":
        model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=logdir)

    episodes = 100
    for ep in range(1, episodes+1):
        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=policy)  # noqa: E501
        model.save(f"{models_dir}/{TIMESTEPS*ep}")
    env.close()


def _test_model(df_add, filename, n_zip, policy="A2C", id="TradingEnv"):
    env = gym.make(
        id,
        name="ADVANC",
        df=df_add["2020-01-01":],
        positions=[0, 0.25, 0.5, 0.75, 1],
        # positions=[0, 1],
        portfolio_initial_value=1000000,
        initial_position=0,
    )
    env.reset()

    models_dir = f"models/{filename}"
    model_path = f"{models_dir}/{n_zip}.zip"

    if policy == "A2C":
        model = A2C.load(model_path, env=env)

    if policy == "PPO":
        model = PPO.load(model_path, env=env)

    episodes = 50
    for ep in range(1, episodes+1):
        obs, info = env.reset()
        done, truncated = False, False
        while not done and not truncated:
            obs = obs[np.newaxis, ...]
            action, _ = model.predict(obs)
            obs, reward, done, truncated, info = env.step(action[0])
            # print('reward =', reward)
        print('reward =', info['reward'], '\nportfolio_valuation(%) =',
              (info['portfolio_valuation']/1000000 - 1) * 100)  # noqa: E501
    env.close()


if __name__ == "__main__":
    df = _read_data()
    df_add = _add_feature(df)
    # _train_model(df_add, "PPO")
    _test_model(df_add, filename="TradingEnv-PPO-1692782276", n_zip="100000", policy="PPO")  # noqa: E501
