"""
Custom exceptions for the Python Trader Backtest framework.

This module defines custom exception classes for better error handling
and more informative error messages throughout the backtesting system.
"""


class BacktestError(Exception):
    """Base exception for all backtesting-related errors."""
    pass


class DataFetchError(BacktestError):
    """Raised when there's an error fetching market data from external sources."""
    pass


class InsufficientFundsError(BacktestError):
    """Raised when a trader doesn't have enough liquid funds to complete a transaction."""
    pass


class InsufficientSharesError(BacktestError):
    """Raised when a trader doesn't have enough shares to complete a sell order."""
    pass


class InvalidParameterError(BacktestError):
    """Raised when invalid parameters are provided to a class or method."""
    pass


class MarketClosedError(BacktestError):
    """Raised when attempting to trade while the market is closed."""
    pass


class InvalidTickerError(BacktestError):
    """Raised when an invalid or unknown ticker symbol is provided."""
    pass

# Made with Bob
