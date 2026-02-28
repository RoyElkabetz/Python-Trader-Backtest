"""
Python Trader Backtest - A comprehensive backtesting framework for trading strategies.

This package provides tools for simulating trading strategies with realistic
transaction costs, portfolio rebalancing, and performance analytics.
"""

from .markets import Market
from .brokers import Broker
from .traders import Trader
from .position import Position
from .simulators import base_simulator, multi_period_simulator
from .optimization import BayesianOptimizer, calculate_ratios, normalize, composite_objective
from .logging_config import setup_logging, get_logger

__version__ = "2.0.0"

__all__ = [
    # Core classes
    "Market",
    "Broker",
    "Trader",
    "Position",
    
    # Simulators
    "base_simulator",
    "multi_period_simulator",
    
    # Optimization
    "BayesianOptimizer",
    "calculate_ratios",
    "normalize",
    "composite_objective",
    
    # Logging
    "setup_logging",
    "get_logger",
]

# Made with Bob
