import torch as th
from stable_baselines3.common.callbacks import EvalCallback

from stock_trading_env.single_stock_env import SingleStockTradingEnv  # noqa
from utils.enviroments import add_metric, env_create, model_create
from utils.preprocess import preprocess_dataframe, train_test_dataframe

# prepare data
df = preprocess_dataframe(
    path="examples/data/AAPL.csv",
    indicators=True,
)

train_df, test_df = train_test_dataframe(df, train_test_split=0.7)

# create env_train
env_train = env_create(
    "single-stock-v1",
    name="AAPL_1",
    df=train_df,
    windows=1,
    positions=[0, 1],
    initial_position=0,
    trading_fees=0,
    portfolio_initial_value=100000,
    max_episode_duration="max",
    verbose=1,
)


env_train = add_metric(env_train)

# define model
policy_kwargs = dict(activation_fn=th.nn.ReLU, net_arch=[256, 256])
model = model_create(
    env=env_train,
    n_steps=2048,
    batch_size=64,
    gae_lambda=0.98,
    gamma=0.999,
    n_epochs=4,
    ent_coef=0.01,
    verbose=0,
    policy_kwargs=policy_kwargs,
    tensorboard_log="./PPO_tensorboard",
)

# create callback
eval_callback = EvalCallback(
    env_train,
    best_model_save_path="./models/",
    log_path="./logs/",
    eval_freq=1000,
    deterministic=False,
    render=False,
)

# training
model.learn(
    total_timesteps=10000, callback=eval_callback, tb_log_name="PPO", progress_bar=True
)
