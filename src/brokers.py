import numpy as np
import pandas as pd
from markets import Market


class Broker:
    def __init__(self, buy_fee: float, min_buy_fee: float,
                 sell_fee: float, min_sell_fee: float, tax: float, my_market: Market):
        self.my_market = my_market
        self.buy_fee = buy_fee
        self.min_buy_fee = min_buy_fee
        self.sell_fee = sell_fee
        self.min_sell_fee = min_sell_fee
        self.tax = tax
        self.pending_buys = []
        self.pending_sells = []

    def buy_now(self, ticker, units):
        # check stock price
        price = self.my_market.get_stock_data(ticker, 'Open')
        total_price = price * units

        # get stocks
        stock = self.my_market.get_stock_data(ticker, 'all')
        stocks = [stock] * units

        # compute the buying fee
        fee = self.buy_fee * total_price
        if fee < self.min_buy_fee:
            fee = self.min_buy_fee

        return stocks, total_price, fee

    def sell_now(self, ticker, stocks):
        # check stock price
        current_price = self.my_market.get_stock_data(ticker, 'Open')
        current_total_price = current_price * len(stocks)

        # compute the stocks value
        stocks_value = 0
        for stock in stocks:
            stocks_value += stock['Open'].values[0]

        # compute the sell fee with respect to current stock price and tax
        sell_fee = current_total_price * self.sell_fee
        tax = np.max([0, current_total_price - stocks_value])

        return current_total_price, sell_fee, tax

    def add_buy(self):
        pass

    def add_sell(self):
        pass

    def execute_buys(self):
        pass

    def execute_sells(self):
        pass

    def charge_trader(self):
        pass
