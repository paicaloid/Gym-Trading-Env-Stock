from typing import Callable, Optional, Union

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from .utils.history import History
from .utils.portfolio import SimplePortfolio


def basic_reward_function(history: History):
    return np.log(
        history["portfolio_valuation", -1] / history["portfolio_valuation", -2]
    )


class SingleStockTradingEnv(gym.Env):
    metadata = {"render_modes": ["logs"]}

    def __init__(
        self,
        df: pd.DataFrame,
        positions: list[Union[int, float]] = [0, 1],
        dynamic_feature_functions: list[Callable] = [],
        reward_function: Callable = basic_reward_function,
        windows: Optional[int] = None,
        trading_fees: float = 0.0,
        portfolio_initial_value: int = 1000,
        initial_position: Union[int, float, str] = "random",
        max_episode_duration: Union[int, str] = "max",
        verbose: int = 1,
        name: str = "StockTradingEnv",
        render_mode: Optional[str] = "logs",
    ) -> None:
        self.max_episode_duration = max_episode_duration
        self.name = name
        self.verbose = verbose
        self.trading_fees = trading_fees
        self.positions = positions
        self.dynamic_feature_functions = dynamic_feature_functions
        self.reward_function = reward_function
        self.windows = windows
        self.portfolio_initial_value = float(portfolio_initial_value)
        self.initial_position = initial_position

        assert (
            self.initial_position in self.positions
            or self.initial_position == "random"
            or self.initial_position in self.positions
        ), (
            "The 'initial_position' parameter must be 'random' or a position "
            "mentionned in the 'position' (default is [0, 1]) parameter."
        )
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self._set_df(df)
        self.action_space = spaces.Discrete(len(positions))
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=[self._nb_features])
        if self.windows is not None:
            self.observation_space = spaces.Box(
                -np.inf, np.inf, shape=[self.windows, self._nb_features]
            )

    def _set_df(self, df: pd.DataFrame) -> None:
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

    def _get_price(self, delta=0):
        return self._price_array[self._idx + delta]

    def _get_obs(self):
        for i, dff in enumerate(self.dynamic_feature_functions):
            self._obs_array[self._idx, self._nb_static_features + i] = dff(
                self.historical_info
            )

        if self.windows is None:
            _step_index = self._idx
        else:
            _step_index = np.arange(self._idx + 1 - self.windows, self._idx + 1)
        return self._obs_array[_step_index]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self._step = 0
        if self.initial_position == "random":
            self._position = np.random.choice(self.positions)
        else:
            self._position = self.initial_position

        self._idx = 0
        if self.windows is not None:
            self._idx = self.windows - 1
        if self.max_episode_duration != "max":
            self._idx = np.random.randint(
                low=self._idx, high=len(self.df) - self.max_episode_duration - self._idx
            )

        self._portfolio = SimplePortfolio(
            position=self._position,
            init_cash=self.portfolio_initial_value,
            current_price=self._get_price(),
            size_granularity=100,
        )

        self.historical_info = History(max_size=len(self.df))
        self.historical_info.set(
            idx=self._idx,
            step=self._step,
            date=self.df.index.values[self._idx],
            position_index=self.positions.index(self._position),
            position=self._position,
            data=dict(zip(self._info_columns, self._info_array[self._idx])),
            portfolio_valuation=self.portfolio_initial_value,
            reward=0,
        )

        return self._get_obs(), self.historical_info[0]

    def _take_action(self, position):
        if position != self._position:
            self._portfolio.trade_to_position(
                position=position,
                price=self._get_price(),
                trading_fees=self.trading_fees,
            )
            self._position = position

    def step(self, position_index=None):
        if position_index is not None:
            self._take_action(position=self.positions[position_index])

        self._idx += 1
        self._step += 1

        price = self._get_price()
        portfolio_value = self._portfolio.get_port_value(price=price)

        done, truncated = False, False

        if portfolio_value <= 0:
            done = True
        if self._idx >= len(self.df) - 1:
            truncated = True
        if (
            isinstance(self.max_episode_duration, int)
            and self._step >= self.max_episode_duration - 1
        ):
            truncated = True

        self.historical_info.add(
            idx=self._idx,
            step=self._step,
            date=self.df.index.values[self._idx],
            position_index=position_index,
            position=self._position,
            data=dict(zip(self._info_columns, self._info_array[self._idx])),
            portfolio_valuation=portfolio_value,
            reward=0,
        )

        if not done:
            reward = self.reward_function(self.historical_info)
            self.historical_info["reward", -1] = reward

        if done or truncated:
            pass

        return (
            self._get_obs(),
            self.historical_info["reward", -1],
            done,
            truncated,
            self.historical_info[-1],
        )
