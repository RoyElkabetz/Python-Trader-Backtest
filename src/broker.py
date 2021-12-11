import numpy as np
import pandas as pd


class Broker:
    def __init__(self, periodic_fee: tuple, buy_fee: float,
                 sell_fee: float, time_delay: float, tax: float):
        self.periodic_fee = periodic_fee
        self.buy_fee = buy_fee
        self.sell_fee = sell_fee
        self.time_delay = time_delay
        self.tax = tax
        self.pending_buys = []
        self.pending_sells = []

    def buy_now(self, ticker, units):
        pass

    def sell_now(self, stocks):
        pass

    def add_buy(self, ticker, units, price, time_window, buy_now=False):

        pass

    def add_sell(self):
        pass

    def execute_buys(self):
        pass

    def execute_sells(self):
        pass

    def charge_trader(self):
        pass
