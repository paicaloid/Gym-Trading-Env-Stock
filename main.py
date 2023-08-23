import gym
import numpy as np
from agent import Agent
import talib as ta
import tensorflow as tf
from datetime import datetime
from agent import Agent
import pandas as pd
import numpy as np
import gymnasium as gym
import datetime
import gym_trading_env
from gym_trading_env.downloader import download
from gym_trading_env.renderer import Renderer
import matplotlib.pyplot as plt
from src.stock_trading_env.single_stock_env import SingleStockTradingEnv
from stable_baselines3 import PPO


def plot_learning_curve(x, scores):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
    plt.plot(x, running_avg)
    plt.xlabel('episodes')
    plt.ylabel("average score")
    plt.title('Running average scores')
    plt.show()

df = pd.read_csv("examples/data/AAPL.csv", parse_dates=["date"], index_col="date")
df.sort_index(inplace=True)
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)

df_add = df.copy()
df_add["feature_close"] = df_add["close"].pct_change()
df_add["feature_open"] = df_add["open"] / df_add["close"]
df_add["feature_high"] = df_add["high"] / df_add["close"]
df_add["feature_low"] = df_add["low"] / df_add["close"]

# def reward_function(history):
#     return np.log(history["portfolio_valuation", -1] / history["portfolio_valuation", -2]) #log (p_t / p_t-1 )


env = SingleStockTradingEnv(
    df=df_add["2020-01-01":],
    positions=[0, 0.25, 0.5, 0.75, 1],
    portfolio_initial_value=100000,
    initial_position=0,
)

# env = gym.make(
#         "TradingEnv",
#         name="BTCUSD",
#         df=df,
#         windows=5,
#         positions=[ -1, -0.5, 0, 0.5, 1, 1.5, 2],  # From -1 (=SHORT), to +1 (=LONG)
#         initial_position='random',  #Initial position
#         trading_fees=0.01/100,  # 0.01% per stock buy / sell
#         borrow_interest_rate=0.0003/100,  #per timestep (= 1h here)
#         reward_function=reward_function,
#         portfolio_initial_value=20,  # in FIAT (here, USD)
#         max_episode_duration=500,
#     )

# env.add_metric('Position Changes', lambda history : np.sum(np.diff(history['position']) != 0) )
# env.add_metric('Episode Lenght', lambda history : len(history['position']) )
# env.add_metric('Reward', lambda history : history['reward'][-1])
# env.add_metric('Portfolio Valuation', lambda history : history['portfolio_valuation'][-1])

N = 20
batch_size = 5
n_epochs = 4
alpha = 0.0003
# episodes = 300
score_history = []
learn_iters = 0
avg_score = 0
n_steps = 0


agent = Agent(n_actions=env.action_space.n, batch_size=batch_size,
                alpha=alpha, n_epochs=n_epochs,
                input_dims=env.observation_space.shape)

# done, truncated = False, False
# # observation, info = env.reset()
# print(info)
best_score = env.reward_range[0]
# done, truncated = False, False

score = 0
episode = 100
for i in range(episode):
    print(f"episode : {i}")
    observation, info = env.reset()
    done = False
    truncated = False
    while not done and not truncated:
        action, prob, val = agent.choose_action(observation)
        print("---------------------------------------")
        print(f"episode : {i}")
        print(f"action : {action}")
        observation_, reward, done, truncated, info, port_value = env.step(action)
        print(f"done : {done}")
        print(f"reward : {reward}")
        print(f"port value: {port_value}")
        n_steps += 1
        score += reward
        agent.remember(observation, action, prob, val, reward, done)

        if n_steps % N == 0:
            agent.learn()
            learn_iters += 1
        observation = observation_

    score_history.append(score)
    avg_score = np.mean(score_history[-100:])
    if avg_score > best_score:
        best_score = avg_score
    i += 1

x = [i+1 for i in range(len(score_history))]
print(x)
print(score_history)
plot_learning_curve(x, score_history)
# fig  = plot_learning_curve(x, score_history)
# fig.show()




# running_avg = np.zeros(len(scores))
#     for i in range(len(running_avg)):
#         running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
#     plt.plot(x, running_avg)
#     plt.xlabel('episodes')
#     plt.ylabel("average score")
#     plt.title('Running average scores')
#     plt.show()



# for i in range(episodes):
#     observation, info = env.reset()
#     done = False
#     score = 0
#     while not done:
#         # action, prob, val = agent.choose_action(observation)
#         # action = env.action_space.sample()
#         action, prob, val = agent.choose_action(np.expand_dims(observation, axis=0))
#         # observation_, reward, done, info = env.step(action)
#         observation_, reward, done, truncated, info = env.step(action)

#         print(f"done : {done}")
#         n_steps += 1
#         score += reward
#         agent.remember(observation, action, prob, val, reward, done)
#         print("---------------")
#         if n_steps % N == 0:
#             agent.learn()
#             learn_iters += 1
#         observation = observation_
#         print(i)
#     score_history.append(score)
#     avg_score = np.mean(score_history[-100:])
#     if avg_score > best_score:
#         best_score = avg_score
# x = [i+1 for i in range(len(score_history))]
# plot_learning_curve(x, score_history)


#     # env.save_for_render()

#     # renderer = Renderer(render_logs_dir="render_logs")
#     # renderer.run()

