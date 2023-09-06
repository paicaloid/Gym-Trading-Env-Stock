from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

from gym_trading_env.renderer import Renderer
from stock_trading_env.single_stock_env import SingleStockTradingEnv  # noqa
from utils.enviroments import add_metric, env_create, test_process
from utils.preprocess import preprocess_dataframe, train_test_dataframe

df = preprocess_dataframe(
    path="examples/data/AAPL.csv",
    indicators=True,
)

train_df, test_df = train_test_dataframe(df, train_test_split=0.7)

# create env_test
env_test = env_create(
    "single-stock-v1",
    name="AAPL_1",
    df=test_df,
    windows=1,
    positions=[0, 1],
    initial_position=0,
    trading_fees=0,
    portfolio_initial_value=100000,
    max_episode_duration="max",
    verbose=1,
)

env_test = add_metric(env_test)

# load model
model = PPO.load("models/best_model.zip")

# evaluate policy
for rep in range(5):
    print(f"\nRepetition {rep}")
    mean_reward, std_reward = evaluate_policy(
        model, env_test, n_eval_episodes=10, deterministic=False
    )
    print(f"Eval reward: {mean_reward} (+/-{std_reward})")


test_process(env_test, model)


env_test.save_for_render()

renderer = Renderer(render_logs_dir="render_logs")
renderer.run()
