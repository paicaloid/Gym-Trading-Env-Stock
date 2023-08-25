import pandas as pd

from src.gym_trading_env.environments import TradingEnv

# Import your datas
df = pd.read_csv(
    "examples/data/BTC_USD-Hourly.csv", parse_dates=["date"], index_col="date"
)
df.sort_index(inplace=True)
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)

# Generating features
# WARNING : the column names need to contain keyword 'feature' !
df["feature_close"] = df["close"].pct_change()
df["feature_open"] = df["open"] / df["close"]
df["feature_high"] = df["high"] / df["close"]
df["feature_low"] = df["low"] / df["close"]
df["feature_volume"] = df["Volume USD"] / df["Volume USD"].rolling(7 * 24).max()
df.dropna(inplace=True)

env = TradingEnv(
    df=df,
    positions=[0, 0.25, 0.5, 0.75, 1],
    portfolio_initial_value=1000000,
    initial_position=0,
)

observation, info = env.reset()
print(observation)
print(info)

action = env.action_space.sample()
observation, reward, done, truncated, info = env.step(action)

print(info)
