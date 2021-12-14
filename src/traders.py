import numpy as np
import pandas as pd
from datetime import date
from markets import Market
from brokers import Broker


class Trader:
    def __init__(self, liquid, balance_liquid_lim, balance_period, my_broker: Broker, my_market: Market):
        self.liquid = liquid
        self.balance_liquid_lim = balance_liquid_lim
        self.balance_period = balance_period
        self.my_broker = my_broker
        self.my_market = my_market
        self.portfolio = {}
        self.portfolio_state = {}
        self.portfolio_buy_value = 0
        self.portfolio_current_value = 0
        self.portfolio_profit = 0
        self.fees_and_tax = 0

        self.liquid_history = []
        self.profit_history = []
        self.portfolio_value_history = []
        self.date_history = []
        self.sell_fee_history = []
        self.buy_fee_history = []
        self.tax_history = []

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
            stocks, total_price, fee = self.my_broker.buy_now(ticker, units)
            self.buy_fee_history.append(fee)

            # pay price
            self.liquid -= total_price

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
            money, fee, tax = self.my_broker.sell_now(ticker, stocks_to_sell)
            self.sell_fee_history.append(fee)
            self.tax_history.append(tax)

            # add money to liquid
            self.liquid += money

            # subtract fee and tax
            self.liquid -= fee
            self.liquid -= tax

            return True
        else:
            print(f'The trader dont have enough {ticker} stocks to complete the trade.')
            return False

    def update(self):
        # update the portfolio state with market's current prices
        self.portfolio_buy_value = 0
        self.portfolio_current_value = 0
        self.fees_and_tax = np.sum(self.buy_fee_history) + np.sum(self.sell_fee_history) + np.sum(self.tax_history)

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
            self.portfolio_state[ticker]['sign'] = np.sign(self.portfolio_state[ticker]['current value'] -
                                                           self.portfolio_state[ticker]['buy value'])
        # compute portfolio profit
        self.portfolio_profit = self.portfolio_current_value - self.portfolio_buy_value - self.fees_and_tax

    def step(self, last_date):
        # update portfolio state and values
        self.update()

        # save history
        self.liquid_history.append(self.liquid)
        self.profit_history.append(self.portfolio_profit)
        self.portfolio_value_history.append(self.portfolio_current_value)
        self.date_history.append(last_date)

    def balance(self, tickers, percentages):
        tickers = np.array(tickers)
        percentages = np.array(percentages)

        # check if the portfolio is balanced
        if self.is_balanced(tickers, percentages):
            return

        # get ticker information
        portfolio_std = []
        owned_units = np.zeros(len(tickers), dtype=np.int)
        market_value = np.zeros(len(tickers), dtype=np.float)
        owned_value = np.zeros(len(tickers), dtype=np.float)
        owned_percentage = np.zeros(len(tickers), dtype=np.float)
        predicted_tax = np.zeros(len(tickers), dtype=np.float)
        stocks_buy_value = {}

        for i, ticker in enumerate(tickers):
            owned_units[i] = self.portfolio_state[ticker]['units']
            market_value[i] = self.my_market.get_stock_data(ticker, 'Open')
            owned_value[i] = owned_units[i] * market_value[i]
            owned_percentage[i] = self.portfolio_state[ticker]['percent']
            stocks_buy_value[ticker] = []
            for stock in self.portfolio[ticker]:
                stocks_buy_value[ticker].append(stock['Open'].values[0])

        # compute tax if balance to the mean up to the margin (worst case)
        mean_balance = np.mean(owned_value)
        units_to_mean = np.array((mean_balance - owned_value) / market_value, dtype=np.int)  # round to closest integer
        for i, ticker in enumerate(tickers):
            if units_to_mean[i] > 0:
                predicted_tax[i] = (np.sum(stocks_buy_value[ticker][:units_to_mean[i]]) - units_to_mean[i] * market_value[i]) * self.my_broker.tax
        predicted_tax = predicted_tax * (predicted_tax > 0)
        predicted_fee = np.sum(units_to_mean * (units_to_mean > 0)) * self.my_broker.sell_fee + np.sum(-units_to_mean * (units_to_mean < 0)) * self.my_broker.buy_fee
        usable_liquid = self.liquid + np.sum(owned_value) - np.sum(predicted_tax) - predicted_fee

        # compute the balance to the maximum mean
        value_per_ticker = usable_liquid / len(tickers)
        units_of_balanced = np.array(value_per_ticker / market_value, dtype=np.int)
        units_to_max = units_of_balanced - owned_units

        # recompute the usable liquid
        predicted_max_fee = np.sum(units_to_max * (units_to_max > 0)) * self.my_broker.buy_fee + np.sum(
            -units_to_max * (units_to_max < 0)) * self.my_broker.sell_fee
        usable_liquid -= predicted_max_fee

        # recompute the balance to the maximum mean
        value_per_ticker = usable_liquid / len(tickers)
        units_of_balanced = np.array(value_per_ticker / market_value, dtype=np.int)
        units_to_max = units_of_balanced - owned_units

        # execute balance
        for i, ticker in enumerate(tickers):
            if units_to_max[i] > 0:
                self.buy(ticker, units_to_max[i])
            if units_to_max[i] < 0:
                self.sell(ticker, -units_to_max[i])

    def is_balanced(self, tickers, percentages):
        assert np.sum(percentages) == 1 and len(tickers) == len(percentages)

        owned_units = np.zeros(len(tickers), dtype=np.int)
        market_value = np.zeros(len(tickers), dtype=np.float)
        owned_value = np.zeros(len(tickers), dtype=np.float)

        for i, ticker in enumerate(tickers):
            owned_units[i] = self.portfolio_state[ticker]['units']
            market_value[i] = self.my_market.get_stock_data(ticker, 'Open')
            owned_value[i] = owned_units[i] * market_value[i]

        margin = np.max(market_value) / 2
        mean_value = np.mean(owned_value)
        if np.max(np.abs(owned_value - mean_value)) > margin:
            return False

        else:
            # the portfolio is balanced
            return True

    def mean_balance(self):
        pass

    def balance2(self, tickers, percentages, tries=10):
        tickers = np.array(tickers)
        percentages = np.array(percentages)

        if self.is_balanced(tickers, percentages):
            return
        else:
            portfolio_std = []

            stock_current_owned_value = np.zeros(len(tickers))
            stock_owned_units = np.zeros(len(tickers))
            stock_market_value = np.zeros(len(tickers))
            stock_current_percentage = np.zeros(len(tickers))

            counter = 0
            std = np.inf
            done = self.is_balanced(tickers, percentages)

            while counter < tries and not done and std > 0.01:
                counter += 1

                # sort and unpack portfolio by current value
                for i, ticker in enumerate(tickers):
                    market_value = self.my_market.get_stock_data(ticker, 'Open')
                    units = self.portfolio_state[ticker]['units']
                    owned_value = self.portfolio_state[ticker]['current value']
                    percent = self.portfolio_state[ticker]['percent']

                    stock_market_value[i] = market_value
                    stock_owned_units[i] = units
                    stock_current_owned_value[i] = owned_value
                    stock_current_percentage[i] = percent

                # compute std
                std = np.std(np.array(percentages) - np.array(stock_current_percentage))
                portfolio_std.append(std)

                # sort by owned value
                sorted_idx = np.argsort(stock_current_owned_value)

                # sort everything
                stock_current_owned_value = stock_current_owned_value[sorted_idx]
                stock_current_percentage = stock_current_percentage[sorted_idx]
                stock_owned_units = stock_owned_units[sorted_idx]
                stock_market_value = stock_market_value[sorted_idx]
                percentages = percentages[sorted_idx]
                tickers = tickers[sorted_idx]

                # check if trader has enough liquid to balance by buying
                if self.liquid > self.balance_liquid_lim:

                    # compute the number of units the trader needs to buy
                    units_to_buy = np.int(np.ceil((percentages[0] - stock_current_percentage[0]) *
                                          self.portfolio_current_value / stock_market_value[0]))
                    # buy
                    bought = self.buy(tickers[0], np.max([1, units_to_buy]))
                    print('buy', tickers[0], units_to_buy)

                else:
                    # compute the number of units the trader needs to sell
                    units_to_sell = np.int(np.ceil((stock_current_percentage[-1] - percentages[-1]) *
                                           self.portfolio_current_value / stock_market_value[-1]))

                    # sell
                    sold = self.sell(tickers[-1], np.max([1, units_to_sell]))
                    print('sell', tickers[-1], units_to_sell)

                # update portfolio
                self.update()

                # check if balanced
                done = self.is_balanced(tickers, percentages)



