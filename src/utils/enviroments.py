import gymnasium as gym

# from src.stock_trading_env.single_stock_env import SingleStockTradingEnv


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
        verbose=1,
    )
    return env


def add_metric():
    pass
