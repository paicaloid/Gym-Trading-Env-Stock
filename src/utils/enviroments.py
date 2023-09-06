import gymnasium as gym
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

# from src.stock_trading_env.single_stock_env import SingleStockTradingEnv


def env_create(
    env_name: str,
    name: str,
    df: pd.DataFrame,
    windows: int,
    positions: list[int],
    initial_position: int,
    trading_fees: float,
    portfolio_initial_value: float,
    max_episode_duration: str,
    verbose: int,
) -> gym.Env:
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
        verbose=verbose,
    )
    return env


def add_metric(env: gym.Env):
    env.add_metric("Reward", lambda history: history["reward"][-1])
    env.add_metric(
        "Portfolio Valuation", lambda history: history["portfolio_valuation"][-1]
    )
    return env


def model_create(
    env: gym.Env,
    n_steps: int,
    batch_size: int,
    gae_lambda: float,
    gamma: float,
    n_epochs: int,
    ent_coef: float,
    verbose: int,
    policy_kwargs: dict,
    tensorboard_log: str,
):
    env = Monitor(env)
    model = PPO(
        "MlpPolicy",
        env,
        n_steps=1024,
        batch_size=64,
        gae_lambda=0.98,
        gamma=0.999,
        n_epochs=4,
        ent_coef=0.01,
        verbose=1,
        policy_kwargs=policy_kwargs,
        tensorboard_log="./PPO_tensorboard/",
    )
    return model


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
