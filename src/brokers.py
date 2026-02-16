import numpy as np
import pandas as pd
from markets import Market


class Broker:
    """A Broker class which mitigate between the Trader and the Market. It execute the trades and collect the fees"""
    def __init__(self, buy_fee: float, min_buy_fee: float,
                 sell_fee: float, min_sell_fee: float, tax: float, my_market: Market):
        self.my_market = my_market
        self.buy_fee = buy_fee / 100.
        self.min_buy_fee = min_buy_fee
        self.sell_fee = sell_fee / 100.
        self.min_sell_fee = min_sell_fee
        self.tax = tax / 100.
        self.pending_buys = []
        self.pending_sells = []

    def buy_now(self, ticker, units):
        """
        Immediate buying execution
        :param ticker: the ticker of the stock (type: str)
        :param units: the amount of units to buy (type: int)
        :return: stocks (type: list[pandas.DataFrame, ...]), total_price (type: float), fee (type: float)
        """
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
        """
        Immediate selling execution
        :param ticker: the ticker of the stock (type: str)
        :param stocks: the amount of units to sell (type: int)
        :return: current_total_price (type: float), fee (type: float), tax (type: float)
        """
        # check stock price
        current_price = self.my_market.get_stock_data(ticker, 'Open')
        current_total_price = current_price * len(stocks)

        # compute the stocks value
        stocks_value = 0
        for stock in stocks:
            stocks_value += stock['Open'].values[0]

        # compute the sell fee with respect to current stock price and tax
        fee = current_total_price * self.sell_fee
        if fee < self.min_sell_fee:
            fee = self.min_sell_fee
        tax = max(0, (current_total_price - stocks_value) * self.tax)

        return current_total_price, fee, tax

