from pydantic import validate_call
import numpy as np
import math

all_symbols = ["AAPL", "ADVANC","B","C","D","E","F","G","H","I","J","K"]


class MultiPortfolio:
    @validate_call
    def __init__(
        self,
        # position: confloat(ge=-1, le=1),
        all_symbols: list,
        init_cash: float,
        size_granularity: float = 100,
        max_money_per_symbol : float = 0.1,
        max_symbol : int = 10,
    ) -> None:
        # init state of port per symbol
        self.position = {key : 0 for key in all_symbols}
        self.size = {key : 0 for key in all_symbols}
        self.buy_price = {key : [] for key in all_symbols}

        self.size_granularity = size_granularity
        self.max_money_per_symbol = max_money_per_symbol * init_cash
        self.max_symbol = max_symbol
        self.cash = init_cash
        self.all_symbols = all_symbols
        self.realize_profit = 0

    def calc_size(
        self,
        cash : float,
        price : float,
        position : float
    ):
        return math.floor(
            (np.abs(position) * cash / price) / self.size_granularity
            ) * self.size_granularity

    def get_port_info(
        self,
        price : dict
    ):
        all_opened = self.get_opened_position()
        symbol_value = sum(
                [self.size[key] * price[key] for key in all_opened]
                )
        realize = self.realize_profit
        self.realize_profit = 0
        return {
            "port_value" :  symbol_value + self.cash,
            "opened_symbol_info" : [self.get_symbol_info(key, price[key]) for key in all_opened],
            "realized_profits" : realize,
            "unrealized_profits" : sum(
                [self.size[key] * (price[key] - self.get_avg(key)) for key in all_opened]
                ),
            "remaining_cash" : self.cash,
            }

    def get_symbol_info(self, symbol : str, price : float):
        return {
            "symbol" : symbol,
            "avg_price" : self.get_avg(symbol),
            "volume" : self.size[symbol],
        }

    def get_avg(self, symbol):
        return sum(self.buy_price[symbol]) / len(self.buy_price[symbol]) \
            if len(self.buy_price[symbol]) > 0 else 0

    def get_opened_position(self):
        return [key for key in self.all_symbols if self.position[key] != 0]

    def check_can_buy(self):
        # check if number of symbol hold is less than max_symbol
        # True -> could buy more
        return len(self.get_opened_position()) < self.max_symbol and self.cash > 0

    def trade_to_position(
        self,
        position : float,
        price : float,
        symbol : str,
        trading_fees : float = 0.000,
    ):
        # find size to trade
        cash_to_trade = self.max_money_per_symbol \
            if self.cash >= self.max_money_per_symbol else self.cash
        size = self.size[symbol]
        to_size = self.calc_size(
            cash=cash_to_trade,
            price=price,
            position=position
        )
        size_to_trade = to_size - size
        if size_to_trade > 0:
            # buy
            # re calc size to trade with trading fees
            size_to_trade = self.calc_size(
                cash=cash_to_trade,
                price=price * (1 + trading_fees),
                position=position
                ) - size
            self.cash = self.cash - (size_to_trade * price * (1 + trading_fees))
            self.buy_price[symbol].append(price * (1 + trading_fees))
        else:
            # sell
            self.cash = self.cash + (np.abs(size_to_trade) * price * (1 - trading_fees))
            self.realize_profit += (np.abs(size_to_trade) * ((price * (1 - trading_fees)) - self.get_avg(symbol)))
            self.buy_price[symbol] = []
        self.size[symbol] += size_to_trade
        self.position[symbol] = position
        return True
