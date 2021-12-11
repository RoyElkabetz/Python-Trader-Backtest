import numpy as np
import pandas as pd
import yfinance as yf
from datetime import date


class Market:
    def __init__(self, stocks: list, start_date: tuple, end_date: tuple, date_format: str = '%Y-%m-%d'):
        self.stocks = [stock.upper() for stock in stocks]
        self.tickers = ' '.join(stocks)
        self.date_format = date_format
        self.start_date = date(*start_date)     # start_date = (Year, Month, Day)
        self.end_date = date(*end_date)         # end_date = (Year, Month, Day)
        self.steps = None
        self.current_idx = None
        self.current_date = None
        self.stocks_data = None

        self.get_data_()

    def get_data_(self):
        try:
            self.stocks_data = yf.download(self.tickers,
                                           start=self.start_date.strftime(self.date_format),
                                           end=self.end_date.strftime(self.date_format))
            self.steps = len(self.stocks_data)
            self.current_idx = 0
            self.current_date = self.stocks_data.index[0].date()

        except Exception as e:
            print(f'A problem occurred in {self.stocks} stocks data download...\n'
                  f'The exception is: {e}')

    def __str__(self):
        return f'Stocks: {self.stocks}\n [+] Start: {self.start_date.strftime(self.date_format)}\n [+] End: ' \
               f'  {self.end_date.strftime(self.date_format)}'

    def get_stock_data(self, stock_name, stock_prm):
        # return a single date single parameter of a single stock
        return self.stocks_data.loc[pd.DatetimeIndex([self.current_date])][stock_prm][stock_name.upper()]

    def step(self):
        if self.current_idx < self.steps:
            self.current_idx += 1
            self.current_date = self.stocks_data.index[self.current_idx].date().strftime(self.date_format)
        else:
            print('You are pointing to the last date in the market data.')

    def get_date_data(self, from_date, as_numpy=False):
        # date format: tuple(yyyy, m, d)
        if self.check_date_(from_date):
            the_date = date(*from_date).strftime(self.date_format)
            the_data = self.stocks_data.loc[pd.DatetimeIndex([the_date])]
            if as_numpy:
                return the_data.columns.values, the_data.values[0]
            else:
                return the_data
        else:
            if as_numpy:
                return [], []
            else:
                return pd.DataFrame()

    def check_date_(self, the_date):
        # date format: tuple(yyyy, m, d)
        return date(*the_date).strftime("%Y-%m-%d") in self.stocks_data.index




