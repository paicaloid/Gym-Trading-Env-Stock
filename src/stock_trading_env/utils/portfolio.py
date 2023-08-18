class SimplePortfolio:
    def __init__(
        self,
        position: float,
        cash: float,
        current_price: float,
    ) -> None:
        self.position = position
        if self.position != 0:
            self.cash = cash - (self.position * current_price)
        else:
            self.cash = cash

    def trade_to_position(
        self,
        position: float,
        price: float,
        trading_fees: float
    ):
        current_position = self.position

        if current_position == 0:
            self.cash = self.cash - (position * price)