from gymnasium.envs.registration import register

register(
    id="single-stock-v0",
    entry_point="src.stock_trading_env.single_stock_env:SingleStockTradingEnv",
    # max_episode_steps=300,
)
