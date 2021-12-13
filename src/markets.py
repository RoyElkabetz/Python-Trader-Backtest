import numpy as np
import copy as cp
import pandas as pd
import yfinance as yf
from datetime import date


class Market:
    def __init__(self, stocks: list, start_date: tuple, end_date: tuple, date_format: str = '%Y-%m-%d'):
        self.tickers = [stock.upper() for stock in stocks]
        self.date_format = date_format
        self.start_date = date(*start_date)     # start_date = (Year, Month, Day)
        self.end_date = date(*end_date)         # end_date = (Year, Month, Day)
        self.steps = None
        self.current_idx = None
        self.current_date = None
        self.stocks_data = {}

        self.get_data_()

    def get_data_(self):
        for ticker in self.tickers:
            try:
                stock_data = yf.download(ticker,
                                         start=self.start_date.strftime(self.date_format),
                                         end=self.end_date.strftime(self.date_format))
                self.stocks_data[ticker] = stock_data
            except Exception as e:
                print(f'A problem occurred in {ticker} stocks data download...\n'
                      f'The exception is: {e}')
        self.steps = len(self.stocks_data[self.tickers[0]])
        self.current_idx = 0
        self.current_date = self.stocks_data[self.tickers[0]].index[0].date()

    def __str__(self):
        return f'Stocks: {self.tickers}\n [+] Start: {self.start_date.strftime(self.date_format)}\n [+] End: ' \
               f'  {self.end_date.strftime(self.date_format)}'

    def get_stock_data(self, ticker, stock_prm):
        if stock_prm == 'all':
            # return a single date all parameters of a single stock (type: DataFrame)
            return self.stocks_data[ticker.upper()].loc[pd.DatetimeIndex([self.current_date])]
        else:
            # return a single date single parameter of a single stock (type: float)
            return self.stocks_data[ticker.upper()].loc[pd.DatetimeIndex([self.current_date])][stock_prm].values[0]

    def step(self):
        # get current date
        previous_date = self.stocks_data[self.tickers[0]].index[self.current_idx].date()

        # step index
        self.current_idx += 1

        # step a single time step forward
        if self.current_idx < self.steps:

            # step date
            self.current_date = self.stocks_data[self.tickers[0]].index[self.current_idx].date().strftime(self.date_format)
            return False, previous_date
        else:
            return True, previous_date

    def get_date_data(self, from_date, as_numpy=False):
        # date format: tuple(yyyy, m, d)
        if self.check_date_(from_date):
            the_date = date(*from_date).strftime(self.date_format)
            the_data = []
            tickers = self.tickers
            if as_numpy:
                for ticker in tickers:
                    the_data.append(self.stocks_data[ticker].loc[pd.DatetimeIndex([the_date])].values[0])
                return tickers, the_data
            else:
                for ticker in tickers:
                    the_data.append(self.stocks_data[ticker].loc[pd.DatetimeIndex([the_date])])
                return tickers, the_data
        else:
            return [], []

    def check_date_(self, the_date):
        # date format: tuple(yyyy, m, d)
        return date(*the_date).strftime("%Y-%m-%d") in self.stocks_data[self.tickers[0]].index




