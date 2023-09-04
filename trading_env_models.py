# *** dont remember add rsi normalize
import pandas as pd
from stable_baselines3 import A2C, PPO
from src.stock_trading_env.single_stock_env import SingleStockTradingEnv  # noqa: F401, E501
from src.gym_trading_env.environments import TradingEnv  # noqa: F401
import gymnasium as gym
import os
import time
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv
from utils.split_data import train_test_split

# from agent import Agent


def _read_data(file="data/ADVANC.csv"):
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
        df=df_add,
        positions=[0, 0.5, 1],
        initial_position='random',
        trading_fees=0,
        # positions=[0, 1],
        max_episode_duration=100,
        portfolio_initial_value=1000000,
        verbose=1,
    )
    # def env_maker(): return env_tmp
    # env = DummyVecEnv([env_maker])
    env.reset()
    # env.add_metric('Position Changes', lambda history : np.sum(np.diff(history['position']) != 0) )
    # env.add_metric('Episode Lenght', lambda history : len(history['position']) )
    # env.add_metric('Reward', lambda history : history['reward'][-1])
    # env.add_metric('Portfolio Valuation', lambda history : history['portfolio_valuation'][-1])
    TIMESTEPS = 200
    env.reset()
    if policy == "A2C":
        model = A2C("MlpPolicy", env, verbose=1, tensorboard_log=logdir)
    elif policy == "PPO":
        model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=logdir)
    episodes = 100
    for ep in range(1, episodes+1):
        # start_time = time.process_time()
        start_time = time.time()
        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=policy)  # noqa: E501
        print("time.time", time.time() - start_time)
        print('time====>', time.process_time()-start_time)
        model.save(f"{models_dir}/{TIMESTEPS*ep}")
        # env.unwrapped.save_for_render(dir="render_logs")
    env.close()


def _test_model(df_add, filename, n_zip, policy="A2C", id="TradingEnv-v2"):
    env = gym.make(
        id,
        name="ADVANC",
        df=df_add,
        positions=[0, 0.5, 1, 1.5],
        # positions=[0, 0.5, 1],
        initial_position='random',
        trading_fees=0,
        # positions=[0, 1],
        portfolio_initial_value=1000000,
        # max_episode_duration=100,

        verbose=1,
    )
    env.reset()
    # env.add_metric('Position Changes', lambda history : np.sum(np.diff(history['position']) != 0) )
    # env.add_metric('Episode Lenght', lambda history : len(history['position']) )
    # env.add_metric('Reward', lambda history : history['reward'][-1])
    # env.add_metric('Portfolio Valuation', lambda history : history['portfolio_valuation'][-1])

    models_dir = f"models/{filename}"
    model_path = f"{models_dir}/{n_zip}"

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
    env.unwrapped.save_for_render(dir="render_logs")
    env.close()


if __name__ == "__main__":
    df = _read_data()
    # df_add = _add_feature(df)
    df_train, df_test = train_test_split(df,
                                         start_train="2020-01-01",
                                         end_train="2021-01-01",
                                         end_test="2022-01-01")
    _train_model(df_train, "PPO")
    # _test_model(df_train, filename="TradingEnv-PPO-1693378336", n_zip="17400", policy="PPO")  # noqa: E501
    # print(df_train)
