# account.py

import pandas as pd


class Account:
    def __init__(self, initial_capital: float = 100000):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.holdings = 0
        self.total = initial_capital
        self.orders = []
        self.position = 0

    def place_order(self, date, order_type, price, amount):
        self.orders.append(
            {"date": date, "order_type": order_type, "price": price, "amount": amount}
        )
        if order_type == "buy":
            self.cash -= price * amount
            self.holdings += price * amount
            self.position += amount
        elif order_type == "sell":
            self.cash += price * amount
            self.holdings -= price * amount
            self.position -= amount
        self.update_total()

    def update_total(self):
        self.total = self.cash + self.holdings

    def current_net_value(self):
        return self.total

    def current_cash(self):
        return self.cash

    def current_position(self):
        return self.position

    def current_percent_gain(self):
        return (self.total / self.initial_capital - 1) * 100

    def get_orders(self):
        return pd.DataFrame(self.orders)
