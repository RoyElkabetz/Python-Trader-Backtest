"""
Portfolio Comparison Demo

This script demonstrates comparing multiple famous portfolio strategies
using the backtesting simulator. It runs simulations for various portfolios
and generates comparison plots.

Usage:
    python examples/portfolio_comparison_demo.py
"""

import sys
import os
# Add parent directory to path to import src modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# from src.main import simulator
from src.simulators import multi_period_simulator
import numpy as np
from src.utils import portfolio_values, yields, fees_and_tax, plot_performance_metrics, print_performance_summary
from src.logging_config import setup_logging


class Portfolio:
    def __init__(self, name, tickers, percentages):
        self.name = name
        self.tickers = tickers
        self.ratios = np.array(percentages) / 100.
        self.traders = None
        assert np.round(np.sum(self.ratios), 5) == 1, f'ratios should sum up to 1, instead the sum was ' \
                                                      f'{np.sum(self.ratios)}'
        assert len(self.ratios) == len(self.tickers), 'lengths of ratios and tickers lists should be the same.'

    def add_traders(self, traders_list):
        self.traders = traders_list


if __name__ == '__main__':

    # Initialize logging system
    setup_logging(
        console_level="INFO",
        file_level="DEBUG",
        log_file="portfolio_optimization.log",
        log_dir="logs",
        enable_console=False,
        enable_file=True,
    )

    # periods to simulation
    periods = [30]

    # Simulator arguments
    liquid = 100000
    sell_strategy = 'LIFO'
    start_date = (2024, 2, 16)
    end_date = (2026, 1, 1)
    buy_fee = 0
    min_buy_fee = 1.
    sell_fee = 0
    min_sell_fee = 1.
    tax = 25.
    verbose = False
    deposit = 0
    deposit_period = 0

    # portfolios to simulate
    portfolio1 = Portfolio('Roy unbalanced',
                           ['AAPL', 'GOOG', 'IBIT', 'META', 'IBM', 'PHO', 'VTI', 'NVDA', 'MSFT'],
                           [16.7, 22.4, 2.5, 11.1, 12.0, 7.4, 9.2, 16.7, 2.0])

    portfolio2 = Portfolio('Roy balanced',
                           ['AAPL', 'GOOG', 'IBIT', 'META', 'IBM', 'PHO', 'VTI', 'NVDA', 'MSFT'],
                           [15, 15, 5, 15, 10, 12.5, 12.5, 10, 5.0])
    
    portfolio3 = Portfolio('Harry Browne',
                           ['VTI', 'GLD', 'TLT', 'SHV'],
                           [25., 25., 25., 25.])
    
    portfolio4 = Portfolio('Warren Buffett',
                           ['SPY', 'BND'],
                           [90., 10.])
    
    

    portfolios_list = [portfolio1, portfolio2, portfolio3, portfolio4]

    # portfolios_list = [portfolio1, portfolio2]

    for i, portfolio in enumerate(portfolios_list):
        # run simulator
        traders_list, market = multi_period_simulator(
            liquid, portfolio.tickers, periods, portfolio.ratios, sell_strategy, start_date,
            end_date, buy_fee, min_buy_fee, sell_fee, min_sell_fee, tax, verbose,
            deposit, deposit_period
            )

        # save results
        portfolio.add_traders(traders_list)

    # Extract traders and names for plotting
    # simulator returns a list of traders (one per period) flatten everything into a single list
    traders_list = []
    portfolio_names = []
    for p in portfolios_list:
        for trader, period in zip(p.traders, periods):
            traders_list.append(trader)
            portfolio_names.append(f"{p.name} - {period}")
    
    # ==================== Plot 1: Portfolio Value History ====================
    portfolio_values(traders_list, portfolio_names, 'Portfolio', use_colors=True)

    # ==================== Plot 2: Yield History ====================
    yields(traders_list, portfolio_names, 'Portfolio', market=None, use_colors=True)

    # ==================== Plot 3: Fees and Tax History ====================
    fees_and_tax(traders_list, portfolio_names, 'Portfolio', use_colors=True)

    # ==================== Plot 4: Performance Metrics Comparison ====================
    plot_performance_metrics(traders_list, portfolio_names, use_colors=True)
    
    # ==================== Print Summary Table ====================
    print_performance_summary(traders_list, portfolio_names, start_date, end_date, liquid, periods[0])



