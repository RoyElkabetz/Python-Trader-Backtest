import pandas as pd
import yfinance as yf
import copy as cp
from datetime import date
from typing import List, Tuple, Dict, Optional, Union, Any
import numpy as np
from .exceptions import DataFetchError, InvalidParameterError
from .logging_config import get_logger

logger = get_logger('markets')


class Market:
    """A Market class which holds the stocks data and dates"""
    def __init__(self, stocks: List[str], start_date: Tuple[int, int, int],
                 end_date: Tuple[int, int, int], date_format: str = '%Y-%m-%d',
                 benchmark_index: str = '^GSPC') -> None:
        """
        Initialize Market with stock data and benchmark index.
        
        Args:
            stocks: List of stock tickers
            start_date: Start date as tuple (year, month, day)
            end_date: End date as tuple (year, month, day)
            date_format: Date format string (default: '%Y-%m-%d')
            benchmark_index: Benchmark index ticker (default: '^GSPC' for S&P 500)
                           Common alternatives: '^DJI' (Dow Jones), '^IXIC' (NASDAQ),
                           '^FTSE' (FTSE 100), '^N225' (Nikkei 225)
        """
        self.tickers = [stock.upper() for stock in stocks]
        self.date_format = date_format
        self.start_date = date(*start_date)     # start_date = (Year, Month, Day)
        self.end_date = date(*end_date)         # end_date = (Year, Month, Day)
        self.index = benchmark_index
        self.common_first_date = None
        self.steps = None
        self.current_idx = None
        self.current_date = None
        self.stocks_data = {}
        self.index_data = None
        self.index_return_percent = None
        self.current_data_cache = {}  # Cache for current day's data

        logger.info(f"Initializing Market with {len(self.tickers)} tickers: {', '.join(self.tickers)}")
        logger.info(f"Date range: {self.start_date} to {self.end_date}")
        logger.info(f"Benchmark index: {self.index}")
        
        self.get_data_()
        if self.current_date != self.stocks_data[self.tickers[0]].index[0].date():
            logger.debug(f"Adjusting start date from {self.start_date} to {self.current_date}")
            self.start_date = self.current_date
            self.get_data_()

        self.get_index()
        logger.info(f"Market initialization complete. Total trading days: {self.steps}")

    def reset(self) -> None:
        """
        Reset the market dates and delete all cached values
        """
        logger.debug(f"Resetting market to initial state (date: {self.common_first_date})")
        self.current_idx = 0
        self.current_date = self.common_first_date
        self.current_data_cache.clear()
        logger.debug("Market reset complete")

    def get_data_(self) -> None:
        """
        Download the data from the Yahoo finance website and saves it as a property
        """
        for ticker in self.tickers:
            try:
                logger.debug(f"Fetching data for {ticker}")
                stock_data = yf.download(ticker,
                                         start=self.start_date.strftime(self.date_format),
                                         end=self.end_date.strftime(self.date_format))
                
                if stock_data.empty:
                    raise DataFetchError(f"No data returned for {ticker}")
                
                self.stocks_data[ticker] = stock_data
                logger.debug(f"Successfully fetched {len(stock_data)} days of data for {ticker}")
            except Exception as e:
                logger.error(f"Failed to fetch data for {ticker}: {e}")
                raise DataFetchError(f"Could not fetch data for {ticker}") from e
        self.steps = len(self.stocks_data[self.tickers[0]])
        self.current_idx = 0

        # get the first common date of all tickers
        self.common_first_date = self.stocks_data[self.tickers[0]].index[0].date()
        for ticker in self.tickers:
            if self.common_first_date < self.stocks_data[ticker].index[0].date():
                self.common_first_date = self.stocks_data[ticker].index[0].date()

        self.current_date = cp.copy(self.common_first_date)

    def get_index(self) -> None:
        """Fetch and process benchmark index data."""
        try:
            logger.debug(f"Fetching benchmark index data for {self.index}")
            index_data = yf.download(self.index,
                                     start=self.start_date.strftime(self.date_format),
                                     end=self.end_date.strftime(self.date_format))
            
            if index_data.empty:
                raise DataFetchError(f"No data returned for benchmark index {self.index}")
            
            self.index_data = index_data

            # compute the percentage index return
            # Extract scalar value properly from pandas
            index_initial_value = self.index_data['Open'].iloc[0]
            if hasattr(index_initial_value, 'item'):
                index_initial_value = index_initial_value.item()
            index_values = self.index_data['Open'].to_numpy()
            self.index_return_percent = (index_values / index_initial_value - 1.) * 100.
            
            logger.debug(f"Successfully fetched benchmark index data for {self.index}")

        except Exception as e:
            logger.error(f"Failed to fetch benchmark index {self.index}: {e}")
            raise DataFetchError(f"Could not fetch benchmark index {self.index}") from e

    def get_stock_data(self, ticker: str, stock_prm: str) -> Union[float, pd.DataFrame]:
        """
        Get a single stock data with caching for performance
        
        Args:
            ticker: The ticker of the stock
            stock_prm: The parameter (e.g., 'Open', 'Close', 'all')
            
        Returns:
            Float value for specific parameter, or DataFrame for 'all'
        """
        ticker = ticker.upper()
        
        # Check cache first
        if ticker not in self.current_data_cache:
            # Convert date object to string for pandas lookup if needed
            date_key = self.current_date if isinstance(self.current_date, str) else self.current_date.strftime(self.date_format)
            self.current_data_cache[ticker] = self.stocks_data[ticker].loc[pd.DatetimeIndex([date_key])]
        
        if stock_prm == 'all':
            # return a single date all parameters of a single stock (type: DataFrame)
            return self.current_data_cache[ticker]
        else:
            # return a single date single parameter of a single stock (type: float)
            value = self.current_data_cache[ticker][stock_prm].values[0]
            # Ensure we return a scalar, not an array
            return value.item() if hasattr(value, 'item') else value

    def step(self) -> Tuple[bool, date]:
        """
        Steps the trading date one step to the future
        
        Returns:
            Tuple of (done, previous_date)
        """
        # get current date
        previous_date = self.stocks_data[self.tickers[0]].index[self.current_idx].date()

        # step index
        self.current_idx += 1

        # Clear cache when stepping to new date
        self.current_data_cache.clear()

        # step a single time step forward
        if self.current_idx < self.steps:
            # step date - keep as date object, not string
            self.current_date = self.stocks_data[self.tickers[0]].index[self.current_idx].date()
            logger.debug(f"Market stepped to date: {self.current_date} (step {self.current_idx}/{self.steps})")
            return False, previous_date
        else:
            logger.debug(f"Market simulation complete. Final date: {previous_date}")
            return True, previous_date

    def get_date_data(self, from_date: Tuple[int, int, int],
                      as_numpy: bool = False) -> Tuple[List[str], Union[np.ndarray, pd.DataFrame]]:
        """
        Get all the market data from a specific date
        
        Args:
            from_date: The date as tuple (year, month, day)
            as_numpy: If True returns as numpy array, else DataFrame
            
        Returns:
            Tuple of (tickers list, data)
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






