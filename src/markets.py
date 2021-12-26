import pandas as pd
import yfinance as yf
from datetime import date


class Market:
    """A MArket class which holds the stocks data and dates"""
    def __init__(self, stocks: list, start_date: tuple, end_date: tuple, date_format: str = '%Y-%m-%d'):
        self.tickers = [stock.upper() for stock in stocks]
        self.date_format = date_format
        self.start_date = date(*start_date)     # start_date = (Year, Month, Day)
        self.end_date = date(*end_date)         # end_date = (Year, Month, Day)
        self.index = '^GSPC'
        self.steps = None
        self.current_idx = None
        self.current_date = None
        self.stocks_data = {}
        self.index_data = None
        self.index_return_percent = None

        self.get_data_()
        self.get_index()

    def get_data_(self):
        """
        Download the data from the Yahoo finance website and saves it as a property
        :return: None
        """
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

    def get_index(self):
        try:
            index_data = yf.download(self.index,
                                     start=self.start_date.strftime(self.date_format),
                                     end=self.end_date.strftime(self.date_format))
            self.index_data = index_data

            # compute the percentage index return
            index_initial_value = self.index_data.iloc[0]['Open']
            self.index_return_percent = (self.index_data['Open'].to_numpy() / index_initial_value - 1.) * 100.

        except Exception as e:
            print(f'A problem occurred in {self.index} stocks data download...\n'
                  f'The exception is: {e}')

    def get_stock_data(self, ticker, stock_prm):
        """
        Get a single stock data
        :param ticker: the ticker of the stock
        :param stock_prm: the parameter, i.e. Open, Close, ...
        :return: the data (type: float)
        """
        if stock_prm == 'all':
            # return a single date all parameters of a single stock (type: DataFrame)
            return self.stocks_data[ticker.upper()].loc[pd.DatetimeIndex([self.current_date])]
        else:
            # return a single date single parameter of a single stock (type: float)
            return self.stocks_data[ticker.upper()].loc[pd.DatetimeIndex([self.current_date])][stock_prm].values[0]

    def step(self):
        """
        Steps the trading date one step to the future
        :return: the previous date (datetime.date)
        """
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
        """
        Get all the market data from a specific date
        :param from_date: the date (type: tuple(yyyy, m, d))
        :param as_numpy: if True returns as numpy, if False return a pandas.DataFrame (default: False)
        :return: lists of the tickers and the data
        """
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
        """
        Checks if the given date was a trading date in the current market data
        :param the_date: the date (type: tuple(yyyy, m, d))
        :return: Boolean
        """
        return date(*the_date).strftime("%Y-%m-%d") in self.stocks_data[self.tickers[0]].index






