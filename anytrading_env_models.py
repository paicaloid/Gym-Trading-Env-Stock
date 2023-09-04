# *** dont remember add rsi normalize
# *** dont remember forex
import pandas as pd
from stable_baselines3 import A2C, PPO
from src.gym_anytrading_env.stocks_env import StocksEnv
from stable_baselines3.common.vec_env import DummyVecEnv

import os
import time
import numpy as np
from finta import TA


def _read_data(file="ADVANC_anytrading.csv"):
    df = pd.read_csv(file, parse_dates=['datetime'], index_col="datetime")
    df.sort_index(inplace=True)
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    return df


def _add_feature(df):
    df_add = df.copy()
    df_add["feature_close"] = df_add["Close"].pct_change()
    df_add["feature_open"] = df_add["Open"] / df_add["Close"]
    df_add["feature_high"] = df_add["High"] / df_add["Close"]
    df_add["feature_low"] = df_add["Low"] / df_add["Close"]

    df_add['feature_rsi_30'] = TA.RSI(df_add, 30)
    df_add['feature_rsi_70'] = TA.RSI(df_add, 70)
    df_add.dropna(axis=0, inplace=True)
    return df_add


def _train_model(df_add, policy="A2C", id="any-trading-v0"):
    models_dir = f"models/{id}-{policy}-{int(time.time())}"
    logdir = f"logs/{id}-{policy}-{int(time.time())}"

    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    if not os.path.exists(logdir):
        os.makedirs(logdir)

    env_tmp = StocksEnv(
        df=df_add,
        window_size=10,
        frame_bound=(10, len(df_add)),
    )
    env_maker = lambda: env_tmp  # noqa: E731
    env = DummyVecEnv([env_maker])
    TIMESTEPS = 10000

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


def _test_model(df_add, filename, n_zip, policy="A2C", id="any-trading-v0"):
    env = StocksEnv(
        df=df_add,
        window_size=10,
        frame_bound=(10, len(df_add)),
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
        done = False
        while not done:
            obs = obs[np.newaxis, ...]
            action, _ = model.predict(obs)
            obs, reward, done, truncated, info = env.step(action[0])
        print("episode =", ep, "info =", info)
    env.close()


if __name__ == "__main__":
    df = _read_data("ADVANC_anytrading.csv")
    df_add = _add_feature(df)
    # _train_model(df_add, "PPO", id="any-trading-v0")
    _test_model(df, filename="any-trading-v0-PPO-1692871848", n_zip="1000000", policy="PPO")  # noqa: E501
