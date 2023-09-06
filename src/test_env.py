import numpy as np
from stable_baselines3.common.env_checker import check_env

from stock_trading_env.single_stock_env import SingleStockTradingEnv
from utils import example_dataframe


def _test_add_metric(env: SingleStockTradingEnv):
    env.add_metric(
        "Position Changes", lambda history: np.sum(np.diff(history["position"]) != 0)
    )
    env.add_metric("Episode Lenght", lambda history: len(history["position"]))
    env.add_metric("Reward", lambda history: history["reward"][-1])
    env.add_metric(
        "Portfolio Valuation", lambda history: history["portfolio_valuation"][-1]
    )

    return env


df = example_dataframe()

env = SingleStockTradingEnv(
    df=df,
    portfolio_initial_value=100000,
)

env = _test_add_metric(env)

observation, info = env.reset()
done = False
truncated = False

check_env(env, warn=True)


while not done and not truncated:
    action = env.action_space.sample()
    observation_, reward, done, truncated, info = env.step(action)
