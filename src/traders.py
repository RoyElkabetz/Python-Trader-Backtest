import numpy as np
import pandas as pd
from datetime import date
from markets import Market
from brokers import Broker


class Trader:
    def __init__(self, liquid, balance_period, my_broker: Broker, my_market: Market):
        self.liquid = liquid
        self.balance_period = balance_period
        self.my_broker = my_broker
        self.my_market = my_market
        self.portfolio = {}
        self.portfolio_state = {}
        self.portfolio_buy_value = 0
        self.portfolio_current_value = 0
        self.portfolio_profit = 0

        self.liquid_history = []
        self.profit_history = []
        self.portfolio_value_history = []
        self.date_history = []

    def buy(self, ticker, units):
        ticker = ticker.upper()

        # get the stock current price
        price = self.my_market.get_stock_data(ticker, 'Open')

        # verify trader got enough liquid to complete the trade
        if units * price > self.liquid:
            print(f'Trader does not have enough money to complete the {ticker} stock trade.')
            return False

        else:
            # buy the stocks
            stocks, fee = self.my_broker.buy_now(ticker, units)

            # pay fee
            self.liquid -= fee

            # add stocks to portfolio
            if ticker not in self.portfolio:
                self.portfolio[ticker] = []
                self.portfolio_state[ticker] = {'units': 0, 'buy value': 0.0, 'current value': 0.0, 'sign': 0, 'percent': 0.0}

            for stock in stocks:
                self.portfolio[ticker].append(stock)
                self.portfolio_state[ticker]['units'] += 1
                self.portfolio_state[ticker]['buy value'] += price
                self.portfolio_state[ticker]['current value'] += price

            return True

    def sell(self, ticker, units):
        ticker = ticker.upper()

        # check trader got enough stocks to complete the trade
        if self.portfolio_state[ticker]['units'] >= units:
            stocks_to_sell = []

            # remove stocks from portfolio (FIFO)
            for _ in range(units):
                # remove stock
                stock = self.portfolio[ticker].pop(0)

                # get price when bought
                buy_price = stock['Open'].values[0]

                # remove a single unit from portfolio
                self.portfolio_state[ticker]['units'] -= 1

                # remove "buy value" of stock
                self.portfolio_state[ticker]['buy value'] -= buy_price
                stocks_to_sell.append(stock)

            # send stocks to broker and get money back
            money = self.my_broker.sell_now(ticker, stocks_to_sell)

            # add money to liquid
            self.liquid += money

            return True
        else:
            print(f'The trader dont have enough {ticker} stocks to complete the trade.')
            return False

    def update(self):
        # update the portfolio state with market's current prices
        self.portfolio_buy_value = 0
        self.portfolio_current_value = 0

        # update prices for all owned stocks
        for ticker in self.portfolio:
            current_price = self.my_market.get_stock_data(ticker, 'Close')
            own_units = self.portfolio_state[ticker]['units']
            buy_value = self.portfolio_state[ticker]['buy value']
            self.portfolio_state[ticker]['current value'] = own_units * current_price

            # compute portfolio buy value
            self.portfolio_buy_value += buy_value

            # compute portfolio current value
            self.portfolio_current_value += own_units * current_price

        # compute percentage of each stock value from portfolio current value
        for ticker in self.portfolio:
            self.portfolio_state[ticker]['percent'] = self.portfolio_state[ticker]['current value'] / \
                                                      self.portfolio_current_value

            # compute the sign which indicates if the stock is profitable at the moment
            self.portfolio_state[ticker]['sign'] = np.sign(self.portfolio_state[ticker]['buy value'] -
                                                           self.portfolio_state[ticker]['current value'])
        # compute portfolio profit
        self.portfolio_profit = self.portfolio_current_value - self.portfolio_buy_value

    def step(self, last_date):
        # update portfolio state and values
        self.update()

        # save history
        self.liquid_history.append(self.liquid)
        self.profit_history.append(self.portfolio_profit)
        self.portfolio_value_history.append(self.portfolio_current_value)
        self.date_history.append(last_date)

    def balance(self, tickers, percentages):
        if self.is_balanced(tickers, percentages):
            return
        else:
            pass

    def is_balanced(self, tickers, percentages):
        assert np.sum(percentages) == 1 and len(tickers) == len(percentages)

        # check if the portfolio is balanced
        for i, ticker in enumerate(tickers):
            if np.abs(self.portfolio_state['percent'] - percentages[i]) > 0.01:
                return False
        else:
            # the portfolio is balanced
            return True




