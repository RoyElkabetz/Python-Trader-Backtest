import numpy as np
import pandas as pd
from market import Market
from broker import Broker


class Trader:
    def __init__(self, liquid, balance_period, my_broker: Broker, my_market: Market):
        self.liquid = liquid
        self.balance_period = balance_period
        self.my_broker = my_broker
        self.my_market = my_market
        self.portfolio = {}
        self.portfolio_state = {}
        self.portfolio_buy_value = 0
        self.portfolio_profit = 0

    def buy(self, ticker, units):
        # get the stock current price
        price = self.my_market.get_stock_data(ticker, 'Open')

        # verify trader got enough liquid to complete the trade
        if units * price > self.liquid:
            print(f'Trader does not have enough money to complete the {ticker} stock trade.')
            return False

        else:
            # buy the stocks
            stocks = self.my_broker.buy_now(ticker, units)
            if ticker not in self.portfolio:
                self.portfolio[ticker] = []
                self.portfolio_state[ticker] = {'units': 0, 'buy value': 0.0, 'current value': 0.0, 'percent': 0.0}

            # add stocks to portfolio
            for stock in stocks:
                self.portfolio[ticker].append(stock)
                self.portfolio_state[ticker]['units'] += 1
                self.portfolio_state[ticker]['buy value'] += price
                self.portfolio_state[ticker]['current value'] += price

    def sell(self, ticker, units):
        # check trader got enough stocks to complete the trade
        if self.portfolio_state[ticker]['units'] >= units:
            stocks_to_sell = []

            # remove stocks from portfolio (FIFO)
            for _ in range(units):
                stock = self.portfolio[ticker].pop(0)
                stocks_to_sell.append(stock)

            # send stocks to broker
            money = self.my_broker.sell_now(stocks_to_sell)

            # add money to liquid
            self.liquid += money
        else:
            print(f'The trader dont have enough {ticker} stocks to complete the trade.')

    def update(self):
        pass

    def balance(self):
        pass

    def pending_buy(self, stock_name, units, buy_price, time_window):
        pass

    def pending_sell(self, stock_name, units, sell_price, time_window):
        pass
