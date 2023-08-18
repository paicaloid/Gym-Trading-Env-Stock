import gymnasium as gym
from gymnasium import spaces
import pandas as pd
import numpy as np


def basic_reward_function():
    pass


class SingleStockTradingEnv(gym.Env):
    metadata = {"render_modes": ["logs"]}

    def __init__(
        self,
        df: pd.DataFrame,
        positions: list = [0, 1],
        dynamic_feature_functions=[],
        reward_function=basic_reward_function,
        windows=None,
        trading_fees=0,
        portfolio_initial_value=1000,
        initial_position="random",
        max_episode_duration="max",
        verbose=1,
        name="Stock",
        render_mode="logs",
    ):
        self.max_episode_duration = max_episode_duration
        self.name = name
        self.verbose = verbose

        self.positions = positions
        self.dynamic_feature_functions = dynamic_feature_functions
        self.reward_function = reward_function
        self.windows = windows
        self.portfolio_initial_value = float(portfolio_initial_value)
        self.initial_position = initial_position

        assert (
            self.initial_position in self.positions or
            self.initial_position == "random" or
            self.initial_position in self.positions
        ), "The 'initial_position' parameter must be 'random' or a position " \
            "mentionned in the 'position' (default is [0, 1]) parameter."
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self._set_df(df)
        self.action_space = spaces.Discrete(len(positions))
        self.observation_space = spaces.Box(
            -np.inf,
            np.inf,
            shape=[self._nb_features]
        )
        if self.windows is not None:
            self.observation_space = spaces.Box(
                -np.inf,
                np.inf,
                shape=[self.windows, self._nb_features]
            )

    def _set_df(self, df):
        df = df.copy()
        self._features_columns = [col for col in df.columns if "feature" in col]
        self._info_columns = list(
            set(list(df.columns) + ["close"]) - set(self._features_columns)
        )
        self._nb_features = len(self._features_columns)
        self._nb_static_features = self._nb_features

        for i in range(len(self.dynamic_feature_functions)):
            df[f"dynamic_feature__{i}"] = 0
            self._features_columns.append(f"dynamic_feature__{i}")
            self._nb_features += 1

        self.df = df
        self._obs_array = np.array(self.df[self._features_columns], dtype=np.float32)
        self._info_array = np.array(self.df[self._info_columns])
        self._price_array = np.array(self.df["close"])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.initial_position == 'random':
            self._position = np.random.choice(self.positions)
        else:
            self._position = self.initial_position

        if self.windows is not None:
            self._idx = self.windows - 1
