import numpy as np
import pandas as pd
from datetime import date
from markets import Market
from brokers import Broker


class Trader:
    """ A Trader class for Backtesting simulation of a periodic balancing strategy for stocks trading"""
    def __init__(self, liquid, balance_liquid_lim, balance_period, broker: Broker, my_market: Market):
        self.liquid = liquid
        self.balance_liquid_lim = balance_liquid_lim
        self.balance_period = balance_period
        self.broker = broker
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
        self.std_history = []
        self.sell_fee_history = []
        self.buy_fee_history = []
        self.tax_history = []

    def buy(self, ticker, units):
        ticker = ticker.upper()

        # get the stock current price
        price = self.my_market.get_stock_data(ticker, 'Open')

        # verify trader got enough liquid to complete the trade
        if units * price > self.liquid:
            print(f'Trader does not have enough liquid money to complete the {ticker} stock trade.')
            return False

        else:
            # buy the stocks
            stocks, total_price, fee = self.broker.buy_now(ticker, units)
            print(ticker, total_price, fee)
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

        # check trader got enough stocks to complete the sell
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
            money, fee, tax = self.broker.sell_now(ticker, stocks_to_sell)
            print(ticker, money, fee, tax)
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

    def balance(self, tickers: list, p=None, verbose=False):
        if p is None:
            p = [1. / len(tickers)] * len(tickers)
        tickers = np.array(tickers, dtype=np.str)
        p = np.array(p, dtype=np.float)

        # check if the portfolio is balanced
        if self.is_balanced(tickers, p=p, verbose=verbose):
            return

        # get tickers information
        owned_units = np.zeros(len(tickers), dtype=np.int)
        market_value = np.zeros(len(tickers), dtype=np.float)
        owned_value = np.zeros(len(tickers), dtype=np.float)
        tax = np.zeros(len(tickers), dtype=np.float)
        max_tax = np.zeros(len(tickers), dtype=np.float)
        stocks_buy_value = {}

        # collect the data
        for i, ticker in enumerate(tickers):
            owned_units[i] = self.portfolio_state[ticker]['units']
            market_value[i] = self.my_market.get_stock_data(ticker, 'Open')
            owned_value[i] = owned_units[i] * market_value[i]
            stocks_buy_value[ticker] = [stock['Open'].values[0] for stock in self.portfolio[ticker]]

        # compute tax for balancing to the mean (worst case)
        mean_balance = np.mean(owned_value)

        # compute the number of units needed to balanced portfolio (buy: positive, sell: negative)
        units_to_mean = np.array(np.round((mean_balance - owned_value) / market_value), dtype=np.int)
        units_to_mean_sign = np.sign(units_to_mean)     # sign
        units_to_mean = np.abs(units_to_mean)           # value

        # compute the tax and total fees for this set of trades
        for i, ticker in enumerate(tickers):
            if units_to_mean_sign[i] < 0:
                total_market_value = units_to_mean[i] * market_value[i]
                total_owned_value = np.sum(stocks_buy_value[ticker][:units_to_mean[i]])
                tax[i] = (total_market_value - total_owned_value) * self.broker.tax

        # drop the negative tax (which comes from selling in loss)
        tax = tax * (tax > 0)

        # compute the fees for trades
        sell_fee = np.max([np.sum(units_to_mean * (units_to_mean_sign > 0)) *
                           self.broker.sell_fee, self.broker.min_sell_fee])
        buy_fee = np.max([np.sum(units_to_mean * (units_to_mean_sign < 0)) *
                          self.broker.buy_fee, self.broker.min_buy_fee])
        total_fee = sell_fee + buy_fee

        # compute the estimated amount of total liquid (trader's portfolio market value + total liquid - tax and fees
        # used for balancing to the mean)
        usable_liquid = self.liquid + np.sum(owned_value) - np.sum(tax) - total_fee

        # compute the units needed for balancing to the maximal weighted mean possible
        value_to_max = usable_liquid * p
        units_of_maxed = np.array(np.round(value_to_max / market_value), dtype=np.int)
        units_to_max = units_of_maxed - owned_units
        units_to_max_sign = np.sign(units_to_max)       # sign
        units_to_max = np.abs(units_to_max)             # value

        # compute the tax for balancing at the maximal mean possible
        for i, ticker in enumerate(tickers):
            if units_to_max_sign[i] < 0:
                total_market_value = units_to_max[i] * market_value[i]
                total_owned_value = np.sum(stocks_buy_value[ticker][:units_to_max[i]])
                max_tax[i] = (total_market_value - total_owned_value) * self.broker.tax

        # drop the negative tax (which comes from selling in loss)
        max_tax = max_tax * (max_tax > 0)

        # compute the fee for balancing at the maximal mean possible
        max_sell_fee = np.max([np.sum(market_value * units_to_max * (units_to_max_sign < 0)) *
                               self.broker.sell_fee, self.broker.min_sell_fee])
        max_buy_fee = np.max([np.sum(market_value * units_to_max * (units_to_max_sign > 0)) *
                              self.broker.buy_fee, self.broker.min_buy_fee])
        max_total_fee = max_sell_fee + max_buy_fee

        # compute the total liquid assuming the trader is balancing to the maximal mean possible
        usable_liquid = self.liquid + np.sum(owned_value) - np.sum(max_tax) - max_total_fee

        # recompute the units needed for balancing to the maximal weighted mean possible
        value_to_max = usable_liquid * p
        units_of_maxed = np.array(value_to_max / market_value, dtype=np.int)
        units_to_max = units_of_maxed - owned_units
        units_to_max_sign = np.sign(units_to_max)       # sign
        units_to_max = np.abs(units_to_max)             # value

        # sort operations such that selling comes before buying
        values_for_execution = units_to_max_sign * units_to_max * market_value
        execution_order = np.argsort(values_for_execution)
        tickers = tickers[execution_order]
        units_to_max_sign = units_to_max_sign[execution_order]
        units_to_max = units_to_max[execution_order]

        if verbose:
            print('\n')
            print('| Liquid: {:14.2f} |'.format(np.round(self.liquid, 2)))
            verbose_str = []
            for ticker in tickers:
                verbose_str.append('| ')
                verbose_str.append(ticker)
                verbose_str.append(': {:14.2f} ')
            verbose_str.append('|')

            print(''.join(verbose_str).format(values_for_execution[execution_order]))

        # execute balance
        for i, ticker in enumerate(tickers):
            if units_to_max_sign[i] > 0:
                self.buy(ticker, units_to_max[i])
            if units_to_max_sign[i] < 0:
                self.sell(ticker, units_to_max[i])

        self.update()
        self.is_balanced(tickers, p=p, verbose=verbose)

    def is_balanced(self, tickers, p=None, verbose=False):
        if p is None:
            p = [1. / len(tickers)] * len(tickers)
        tickers = np.array(tickers, dtype=np.str)
        p = np.array(p, dtype=np.float)

        # compute the owned value per ticker
        owned_units = np.zeros(len(tickers), dtype=np.int)
        market_value = np.zeros(len(tickers), dtype=np.float)

        for i, ticker in enumerate(tickers):
            owned_units[i] = self.portfolio_state[ticker]['units']
            market_value[i] = self.my_market.get_stock_data(ticker, 'Open')
        owned_value = owned_units * market_value

        print('needs fixing... and recomputed for weighted mean')

        # every ticker allowed to be far from the mean by a half of its single unit
        allowed_margin = np.sum(market_value / 2)
        goal_values = np.sum(owned_value) * p
        total_error = np.sum(np.abs(owned_value - goal_values))
        if verbose:
            print('| Current Error: {:10.2f} | Allowed Error: {:10.2f}'
                  .format(np.round(total_error), np.round(allowed_margin)))

        if total_error < allowed_margin:
            # the portfolio is balanced up to the allowed margin
            return True
        else:
            # the portfolio is not balanced
            return False




