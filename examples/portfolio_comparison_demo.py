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
        file_level="INFO",
        log_file="portfolio_comparison_demo.log",
        log_dir="logs",
        enable_console=True,
        enable_file=True,
    )

    # periods to simulation
    periods = [30, 90]

    # Simulator arguments
    liquid = 100000
    sell_strategy = 'LIFO'
    start_date = (2020, 2, 16)
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
    portfolio1 = Portfolio('Ideal Index',
                           ['SPY', 'VBK', 'IUSV', 'VBR', 'VNQ', 'VXUS', 'BSV'],
                           [6.25, 6.25, 9.25, 9.25, 8, 31, 30])
    portfolio2 = Portfolio('Harry Browne',
                           ['VTI', 'GLD', 'TLT', 'SHV'],
                           [25., 25., 25., 25.])
    portfolio3 = Portfolio('Warren Buffett',
                           ['SPY', 'BND'],
                           [90., 10.])
    portfolio4 = Portfolio('All Seasons',
                           ['VTI', 'IEF', 'TLT', 'DBC', 'GLD'],
                           [30., 15., 40., 7.5, 7.5])
    portfolio5 = Portfolio('Stocks 100%',
                           ['VT'],
                           [100.])
    portfolio6 = Portfolio('2nd Grade',
                           ['VTI', 'VXUS', 'BND'],
                           [60., 30., 10.])
    portfolio7 = Portfolio('No Brainer',
                           ['VTI', 'VB', 'VEA', 'BND'],
                           [25., 25., 25., 25.])
    portfolio8 = Portfolio('Three Funds',
                           ['VTI', 'VXUS', 'BND'],
                           [40., 20., 40.])
    portfolio9 = Portfolio('Ivy League',
                           ['VTI', 'VXUS', 'BND', 'VNQ', 'DBC'],
                           [20., 20., 20., 20., 20.])
    portfolio10 = Portfolio('Four Cores',
                            ['VTI', 'VXUS', 'VNQ', 'BND'],
                            [48., 24., 8., 20.])
    portfolio11 = Portfolio('Margarita',
                            ['VTI', 'VXUS', 'BND'],
                            [33.3, 33.3, 33.4])
    portfolio12 = Portfolio('Went Fishing',
                            ['VTI', 'VB', 'VGK', 'VPL', 'VWO', 'SHY', 'TIP', 'JNK', 'VNQ', 'GLTR'],
                            [15., 15., 10., 10., 10., 10., 10., 10., 5., 5.])
    portfolio13 = Portfolio('50/50',
                            ['VT', 'BND'],
                            [50., 50.])
    portfolio14 = Portfolio('Foursquare',
                            ['VTI', 'VXUS', 'BND', 'BNDX'],
                            [25., 25., 25., 25.])
    portfolio15 = Portfolio('Unconventional Success',
                            ['VTI', 'VEA', 'VWO', 'TLT', 'TIP', 'VNQ'],
                            [20., 20., 10., 15., 15., 20.])
    portfolio16 = Portfolio('Dark Chocolate',
                            ['VEA', 'VWO', 'TIP', 'BSV'],
                            [55.25, 9.75, 17.50, 17.50])
    portfolio17 = Portfolio('Cherry Pie',
                            ['TLT', 'VWO', 'GLD', 'VBR', 'VNQ', 'BSV'],
                            [12., 12., 12., 12., 12., 40.])
    portfolio18 = Portfolio('Colliding Markets',
                            ['VTI', 'VEA', 'VWO', 'TLT', 'BNDX', 'TIP', 'VNQ', 'DBC'],
                            [18., 18., 15., 6., 11., 6., 13., 13.])
    portfolio19 = Portfolio('Ultimate Buy & Hold',
                            ['SPY', 'IUSV', 'VB', 'VBR', 'VEA', 'VWO', 'EFV', 'VSS', 'VNQ', 'SHY', 'IEF'],
                            [6., 6., 6., 6., 6., 6., 6., 12., 6., 12., 28.])
    portfolio20 = Portfolio('Coffee Shop',
                            ['SPY', 'VB', 'IUSV', 'VBR', 'VNQ', 'VXUS', 'BND'],
                            [10., 10., 10., 10., 10., 10., 40.])
    portfolio21 = Portfolio('Chicken',
                            ['SPY', 'VB', 'IUSV', 'VBR', 'VNQ', 'VGK', 'VPL', 'VWO', 'SHY'],
                            [15., 5., 10., 10., 5., 5., 5., 5., 40.])
    portfolio22 = Portfolio('Big Bricks',
                            ['SPY', 'VB', 'VTV', 'VBR', 'VNQ', 'VXUS', 'EFV', 'VSS', 'DLS', 'VWO', 'BSV'],
                            [9., 9., 9., 9., 6., 3., 6., 3., 3., 3., 40.])
    portfolio23 = Portfolio('7-12',
                            ['VOO', 'VO', 'VB', 'VEA', 'VWO', 'SHV', 'BND', 'TIP', 'BNDX', 'VNQ', 'DBC', 'IGE'],
                            [8.3, 8.3, 8.3, 8.3, 8.3, 8.3, 8.3, 8.3, 8.4, 8.4, 8.4, 8.4])
    portfolio24 = Portfolio('Thick Tail',
                            ['VBR', 'BSV'],
                            [32., 68.])
    portfolio25 = Portfolio('Talmud',
                            ['VT', 'RWO', 'BND'],
                            [33.3, 33.3, 33.4])

    # uncomment for all portfolios
    # portfolios_list = [portfolio1, portfolio2, portfolio3, portfolio4, portfolio5, portfolio6, portfolio7,
    #                    portfolio8, portfolio9, portfolio10, portfolio11, portfolio12, portfolio13, portfolio14,
    #                    portfolio15, portfolio16, portfolio17, portfolio18, portfolio19, portfolio20, portfolio21,
    #                    portfolio22, portfolio23, portfolio24, portfolio25, portfolio26
    #                    ]

    # # uncomment for the first 16 portfolios
    # portfolios_list = [portfolio1, portfolio2, portfolio3, portfolio4, portfolio5, portfolio6, portfolio7,
    #                    portfolio8, portfolio9, portfolio10, portfolio11, portfolio12, portfolio13, portfolio14,
    #                    portfolio15, portfolio16
    #                    ]

    portfolios_list = [portfolio1, portfolio2, portfolio3, portfolio4, portfolio5, portfolio6, portfolio7]

    # portfolios_list = [portfolio1, portfolio2]

    for i, portfolio in enumerate(portfolios_list):
        # run simulator
        traders_list, market = multi_period_simulator(
            liquid, portfolio.tickers, periods, portfolio.ratios, sell_strategy, start_date,
            end_date, buy_fee, min_buy_fee, sell_fee, min_sell_fee, tax, verbose,
            deposit, deposit_period,
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



