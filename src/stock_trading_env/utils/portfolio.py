from pydantic import validate_call
from pydantic.types import confloat
import numpy as np


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
        if self.position != 0:
            self.size = round(
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
        current_position = self.position

        if current_position == 0:
            self.cash = self.cash - (position * price)