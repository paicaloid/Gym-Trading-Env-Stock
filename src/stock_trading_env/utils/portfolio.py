from pydantic import validate_call
from pydantic.types import confloat
import numpy as np
import math


class SimplePortfolio:
    @validate_call
    def __init__(
        self,
        position: confloat(ge=-1, le=1),
        init_cash: float,
        current_price: float,
        size_granularity: float = 100,
    ) -> None:
        self.position = position
        self.size_granularity = size_granularity
        if self.position != 0:
            self.size = math.floor(
                (np.abs(self.position) * init_cash / current_price) / size_granularity
            ) * size_granularity
        else:
            self.size = 0
        self.cash = init_cash - (self.size * current_price)

    def trade_to_position(
        self,
        position: float,
        price: float,
        trading_fees: float
    ):
        port_value = self.get_port_value(price=price)

        # find size to trade
        size_to_trade = math.floor(
                (np.abs(position) * port_value / price)
                / self.size_granularity
            ) * self.size_granularity - self.size
        if size_to_trade > 0:
            # buy
            size_to_trade = math.floor(
                (np.abs(position) * port_value / (price * (1 + trading_fees)))
                / self.size_granularity
            ) * self.size_granularity - self.size
            self.cash = self.cash - (size_to_trade * price * (1 + trading_fees))
        else:
            # sell
            self.cash = self.cash + (np.abs(size_to_trade) * price * (1 - trading_fees))
        self.size += size_to_trade
        self.position = position

    def get_port_value(
        self,
        price : float
    ):
        return (self.size * price) + self.cash
