import numpy as np
from markets import Market
from brokers import Broker
import copy as cp


class Trader:
    """ A Trader class for Backtesting simulation of a periodic balancing strategy for stocks trading"""
    def __init__(self, liquid, balance_period, broker: Broker, market: Market, verbose=False, sell_strategy='FIFO'):
        self.liquid = liquid
        self.balance_period = balance_period
        self.broker = broker
        self.market = market
        self.verbose = verbose

        assert sell_strategy in ['FIFO', 'LIFO', 'TAX_OPT'], \
            'sell_strategy should be one of the following: "FIFO", "LIFO", "TAX_OPT".'
        self.sell_strategy = sell_strategy

        # Trader's portfolio
        self.portfolio = {}
        self.portfolio_meta = {}
        self.portfolio_primary_value = 0
        self.portfolio_market_value = 0
        self.portfolio_profit = 0
        self.sell_fee = 0
        self.buy_fee = 0
        self.tax = 0
        self.fees_and_tax = 0
        self.usable_liquid = 0
        self.portfolio_initial_value = None

        # Save trading history
        self.liquid_history = []
        self.profit_history = []
        self.portfolio_value_history = []
        self.yield_history = []
        self.date_history = []
        self.error_history = []
        self.sell_fee_history = []
        self.buy_fee_history = []
        self.tax_history = []

    def buy(self, ticker, units):
        """
        This function is used for buying new storks and adding them to the trader's portfolio
        :param ticker: the ticker of the stock
        :param units: number of units to buy
        :return: True / False boolean if the trade is succeed / not
        """
        ticker = ticker.upper()

        # get the stock current price
        price = self.market.get_stock_data(ticker, 'Open')

        # verify trader got enough liquid to complete the trade
        if units * price > self.liquid:
            print(f'\n[+][+] Trader does not have enough liquid money to complete the {ticker} stock trade.\n')
            return False
        else:
            # buy the stocks
            stocks, total_price, fee = self.broker.buy_now(ticker, units)
            self.buy_fee += fee

            # pay price
            self.liquid -= total_price

            # pay fee
            self.liquid -= fee

            # add ticker to portfolio
            if ticker not in self.portfolio:
                self.portfolio[ticker] = []
                self.portfolio_meta[ticker] = {'units': 0, 'sign': 0}

            self.portfolio[ticker] += stocks
            self.portfolio_meta[ticker]['units'] += units
            self.portfolio_primary_value += price

            if self.verbose:
                print('[+] BUY  | Ticker: {:6s} | Units: {:4.0f} | Total price: {:10.2f} | Fee: {:8.2f} |'
                      .format(ticker, units, np.round(total_price, 2), np.round(fee, 2)))

            return True

    def sell(self, ticker, units):
        """
        This function is used for selling stocks from the trader's portfolio
        :param ticker: the ticker of the stock
        :param units: number of units to sell
        :return: True / False boolean if the trade is succeed / not
        """
        ticker = ticker.upper()

        # check trader got enough stocks to complete the sell
        if self.portfolio_meta[ticker]['units'] >= units:
            stocks_to_sell = []

            # remove stocks from portfolio in a FIFO order (first in first out)
            for _ in range(units):

                # remove stock from portfolio and subtract its primary value from the cumulative primary value
                stock = self.portfolio[ticker].pop(0)
                primary_price = stock['Open'].values[0]
                self.portfolio_meta[ticker]['units'] -= 1
                self.portfolio_primary_value -= primary_price
                stocks_to_sell.append(stock)

            # send stocks to broker and collect money
            money, fee, tax = self.broker.sell_now(ticker, stocks_to_sell)
            self.sell_fee += fee
            self.tax += tax

            # update the amount of liquid
            self.liquid += money - fee - tax

            if self.verbose:
                print('[+] SELL | Ticker: {:6s} | Units: {:4.0f} | Total price: {:10.2f} | Fee: {:8.2f} '
                      '| Tax: {:8.2f} |'.format(ticker, units, np.round(money, 2), np.round(fee, 2), np.round(tax, 2)))

            return True
        else:
            print(f'\n[+][+] The trader does not have enough {ticker} units to complete the trade.\n')
            return False

    def update(self):
        """
        Function for updating the portfolio with the current market value of all stocks and computing the total profit
        :return: None
        """
        # update the portfolio market current prices
        self.portfolio_market_value = 0

        # update market prices for all owned stocks
        for ticker in self.portfolio:
            market_price = self.market.get_stock_data(ticker, 'Open')
            units = self.portfolio_meta[ticker]['units']
            self.portfolio_market_value += units * market_price

        # compute portfolio profit
        self.fees_and_tax = np.sum(self.buy_fee_history) + np.sum(self.sell_fee_history) + np.sum(self.tax_history)
        self.portfolio_profit = self.portfolio_market_value - self.portfolio_primary_value - self.fees_and_tax

    def step(self, last_date):
        """
        Step one trading day ahead while updating the portfolio and saving portfolio history data for later analysis
        :param last_date: the current trading date
        :return: None
        """
        # update portfolio
        self.update()
        self.sort_tickers()

        # save trading history
        self.buy_fee_history.append(self.buy_fee)
        self.sell_fee_history.append(self.sell_fee)
        self.tax_history.append(self.tax)
        self.buy_fee = 0
        self.sell_fee = 0
        self.tax = 0
        self.liquid_history.append(self.liquid)
        self.profit_history.append(self.portfolio_profit)  # market value - value when bought - tax and fees
        self.portfolio_value_history.append(self.portfolio_market_value)
        if self.portfolio_initial_value is None:
            self.portfolio_initial_value = cp.copy(self.portfolio_market_value)
        self.yield_history.append((self.portfolio_market_value / self.portfolio_initial_value - 1.) * 100.)
        self.date_history.append(last_date)

    def balance(self, tickers: list, p=None):
        """
        This function balances the trader's portfolio according to a given weight list
        :param tickers: All the tickers in the portfolio (type: list)
        :param p: The weights for balancing with respect to the tickers order (type: list )
        :return: None
        """
        if p is None:
            p = [1. / len(tickers)] * len(tickers)
        tickers = np.array(tickers, dtype=str)
        p = np.array(p, dtype=float)

        if self.verbose:
            print('\n')
            print('|------------------------------------------ BALANCING --------------------------------------------|')

        # get tickers information
        owned_units = np.zeros(len(tickers), dtype=int)
        market_value = np.zeros(len(tickers), dtype=float)
        owned_value = np.zeros(len(tickers), dtype=float)
        tax = np.zeros(len(tickers), dtype=float)
        max_tax = np.zeros(len(tickers), dtype=float)
        stocks_buy_value = {}

        # collect the data
        for i, ticker in enumerate(tickers):
            owned_units[i] = self.portfolio_meta[ticker]['units']
            market_value[i] = self.market.get_stock_data(ticker, 'Open')
            owned_value[i] = owned_units[i] * market_value[i]
            stocks_buy_value[ticker] = [stock['Open'].values[0] for stock in self.portfolio[ticker]]

        # compute tax for balancing to the mean (worst case)
        margin = np.sum(market_value) / 2
        mean_balance = np.mean(owned_value) - margin

        # compute the number of units needed to balanced portfolio (buy: positive, sell: negative)
        units_to_mean = np.array(np.round((mean_balance - owned_value) / market_value), dtype=self.new_method())
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
        sell_fee = np.max([np.sum(market_value * units_to_mean * (units_to_mean_sign < 0)) *
                           self.broker.sell_fee, np.sum(units_to_mean_sign < 0) * self.broker.min_sell_fee])
        buy_fee = np.max([np.sum(market_value * units_to_mean * (units_to_mean_sign > 0)) *
                          self.broker.buy_fee, np.sum(units_to_mean_sign > 0) * self.broker.min_buy_fee])
        total_fee = sell_fee + buy_fee

        # compute the estimated amount of total liquid (trader's portfolio market value + total liquid - tax and fees
        # used for balancing to the mean)
        self.usable_liquid = self.liquid + np.sum(owned_value) - np.sum(tax) - total_fee

        # compute the units needed for balancing to the maximal weighted mean possible
        margins = market_value / 2
        value_to_max = self.usable_liquid * p - margins
        units_of_maxed = np.array(np.round(value_to_max / market_value), dtype=int)
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
                               self.broker.sell_fee, np.sum(units_to_max_sign < 0) * self.broker.min_sell_fee])
        max_buy_fee = np.max([np.sum(market_value * units_to_max * (units_to_max_sign > 0)) *
                              self.broker.buy_fee, np.sum(units_to_max_sign > 0) * self.broker.min_buy_fee])
        max_total_fee = max_sell_fee + max_buy_fee

        # compute the total liquid assuming the trader is balancing to the maximal mean possible
        self.usable_liquid = self.liquid + np.sum(owned_value) - np.sum(max_tax) - max_total_fee

        # recompute the units needed for balancing to the maximal weighted mean possible
        value_to_max = self.usable_liquid * p - margins
        units_of_maxed = np.array(np.round(value_to_max / market_value), dtype=int)
        units_to_max = units_of_maxed - owned_units
        units_to_max_sign = np.sign(units_to_max)       # sign
        units_to_max = np.abs(units_to_max)             # value

        # sort operations such that selling comes before buying
        values_for_execution = units_to_max_sign * units_to_max * market_value
        execution_order = np.argsort(values_for_execution)
        tickers = tickers[execution_order]
        units_to_max_sign = units_to_max_sign[execution_order]
        units_to_max = units_to_max[execution_order]

        if self.verbose:
            print('[+] Liquid: {:14.2f} '.format(np.round(self.liquid, 2)))
            execute_str = ['[+] NEXT ']
            for ticker in tickers:
                execute_str.append('| ')
                execute_str.append(ticker)
                execute_str.append(': {:10.2f} ')
            execute_str.append('|')
            print('|-------------------------------------------------------------------------------------------------|')
            print(''.join(['[+] CURR '] + execute_str[1:]).format(*owned_value[execution_order]))
            print(''.join(['[+] GOAL '] + execute_str[1:]).format(*value_to_max[execution_order]))
            print(''.join(execute_str).format(*values_for_execution[execution_order]))
            print(''.join(['[+] UNIT '] + execute_str[1:]).format(*market_value[execution_order]))
            print('|-------------------------------------------------------------------------------------------------|')

        # execute balance
        for i, ticker in enumerate(tickers):
            if units_to_max_sign[i] > 0:
                self.buy(ticker, units_to_max[i])
            if units_to_max_sign[i] < 0:
                self.sell(ticker, units_to_max[i])

        self.update()
        self.is_balanced(tickers, p=p[execution_order])

        if self.verbose:
            print('|-------------------------------------------------------------------------------------------------|')

    def new_method(self):
        return int

    def is_balanced(self, tickers, p=None):
        """
        A function which checks if the trader's portfolio is in balance
        :param tickers: All the tickers in the portfolio (type: list)
        :param p: The weights for balancing with respect to the tickers order (type: list)
        :return: None
        """
        if p is None:
            p = [1. / len(tickers)] * len(tickers)
        tickers = np.array(tickers, dtype=str)
        p = np.array(p, dtype=float)

        # compute the owned value per ticker
        owned_units = np.zeros(len(tickers), dtype=int)
        market_value = np.zeros(len(tickers), dtype=float)

        for i, ticker in enumerate(tickers):
            owned_units[i] = self.portfolio_meta[ticker]['units']
            market_value[i] = self.market.get_stock_data(ticker, 'Open')
        owned_value = owned_units * market_value

        # compute the half single unit margin error
        margins = market_value / 2
        allowed_margin = np.sum(margins)
        goal_values = self.usable_liquid * p - margins
        total_error = np.sum(np.abs(owned_value - goal_values))
        if self.verbose:
            print('| Current Error: {:10.2f} | Allowed Error: {:10.2f}'
                  .format(np.round(total_error), np.round(allowed_margin)))

        if total_error < allowed_margin:
            # the portfolio is balanced up to the allowed margin
            return True
        else:
            # the portfolio is not balanced
            return False

    def sort_tickers(self):
        """
        Sort stocks in portfolio for each ticker in one of the following orders: FIFO, LIFO, TAX_OPT,
        where TAX_OPT will sort the stocks according to their Opening prices, such that when sold would lead to
        a minimal tax payment.
        :return: None
        """
        # FIFO ordering of portfolio stocks
        if self.sell_strategy == 'FIFO':
            return
        # LIFO ordering of portfolio stocks
        elif self.sell_strategy == 'LIFO':
            stocks_dates = {}
            for ticker in self.portfolio:
                stocks_dates[ticker] = []
                stocks = self.portfolio[ticker]
                for stock in stocks:
                    stocks_dates[ticker].append(stock.index[0])
                order = np.argsort(np.array(stocks_dates[ticker]))
                self.portfolio[ticker] = [self.portfolio[ticker][i] for i in order[::-1]]
        # TAX_OPT ordering of portfolio stocks
        elif self.sell_strategy == 'TAX_OPT':
            stocks_price = {}
            for ticker in self.portfolio:
                stocks_price[ticker] = []
                stocks = self.portfolio[ticker]
                for stock in stocks:
                    stocks_price[ticker].append(stock['Open'].values[0])
                order = np.argsort(np.array(stocks_price[ticker]))
                self.portfolio[ticker] = [self.portfolio[ticker][i] for i in order[::-1]]

    def deposit(self, amount):
        """
        Add money to the traders liquid
        :param amount: the amount of money to deposit
        :return: None
        """
        assert amount > 0, 'Trader can only deposit positive amounts of money.'
        self.liquid += amount

    def withdraw(self, amount):
        """
        Withdraw money from trader's liquid
        :param amount: the amount of money to withdraw
        :return: amount ot 0
        """
        assert amount > 0, 'Trader can only withdraw positive amounts of money.'
        if self.liquid >= amount:
            self.liquid -= amount
            return amount
        else:
            print(f'Trader does not have enough liquid (has {self.liquid} $) to withdraw {amount} $.')
            return 0
