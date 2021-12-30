import matplotlib.pyplot as plt
from main import simulator
import numpy as np
from utils import *
import copy as cp


class Portfolio:
    def __init__(self, name, tickers, percentages):
        self.name = name
        self.tickers = tickers
        self.ratios = np.array(percentages) / 100.
        self.traders = None
        assert np.round(np.sum(self.ratios), 5) == 1, f'ratios should sum up to 1, instead the sum was {np.sum(self.ratios)}'
        assert len(self.ratios) == len(self.tickers), 'lengths of ratios and tickers lists should be the same.'

    def add_traders(self, traders_list):
        self.traders = traders_list


if __name__ == '__main__':
    # periods to simulation
    periods = [30]

    # Simulator arguments
    liquid = 100000
    sell_strategy = 'TAX_OPT'
    start_date = (2020, 1, 1)
    end_date = (2021, 12, 30)
    buy_fee = 0
    min_buy_fee = 1.
    sell_fee = 0
    min_sell_fee = 1.
    tax = 25.
    verbose = False
    plots_normalize = False
    deposit = 0
    deposit_period = 0
    show_plots = False
    return_traders = True

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

    # # uncomment for all portfolios
    # portfolios_list = [portfolio1, portfolio2, portfolio3, portfolio4, portfolio5, portfolio6, portfolio7,
    #                    portfolio8, portfolio9, portfolio10, portfolio11, portfolio12, portfolio13, portfolio14,
    #                    portfolio15, portfolio16, portfolio17, portfolio18, portfolio19, portfolio20, portfolio21,
    #                    portfolio22, portfolio23, portfolio24, portfolio25
    #                    ]

    # # uncomment for the first 16 portfolios
    # portfolios_list = [portfolio1, portfolio2, portfolio3, portfolio4, portfolio5, portfolio6, portfolio7,
    #                    portfolio8, portfolio9, portfolio10, portfolio11, portfolio12, portfolio13, portfolio14,
    #                    portfolio15, portfolio16
    #                    ]

    portfolios_list = [portfolio1, portfolio2, portfolio3, portfolio4, portfolio5, portfolio6, portfolio7]

    for i, portfolio in enumerate(portfolios_list):
        # run simulator
        traders_list = simulator(liquid, portfolio.tickers, periods, portfolio.ratios, sell_strategy, start_date,
                                 end_date, buy_fee, min_buy_fee, sell_fee, min_sell_fee, tax, verbose,
                                 plots_normalize, deposit, deposit_period, show_plots, return_traders)

        # save results
        portfolio.add_traders(traders_list)

    # plot yields
    interval = np.int(len(portfolio.traders[0].date_history) / 10)
    fig, axes = plt.subplots(nrows=1, ncols=1, sharex=True, dpi=150)
    axes.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    axes.xaxis.set_major_locator(mdates.DayLocator(interval=interval))
    axes.set_title('Value history')
    for i, portfolio in enumerate(portfolios_list):
        axes.plot(portfolio.traders[0].date_history, portfolio.traders[0].portfolio_value_history,
                  label=portfolio.name, linewidth=1)

    axes.set_ylabel('USD')
    fig.autofmt_xdate(bottom=0.2, rotation=30, ha='right')
    axes.legend(fontsize=4)
    axes.grid()
    plt.show()

    interval = np.int(len(portfolio.traders[0].date_history) / 10)
    fig, axes = plt.subplots(nrows=1, ncols=1, sharex=True, dpi=150)
    axes.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    axes.xaxis.set_major_locator(mdates.DayLocator(interval=interval))
    axes.set_title('Yield history')
    for i, portfolio in enumerate(portfolios_list):
        axes.plot(portfolio.traders[0].date_history, portfolio.traders[0].yield_history,
                  label=portfolio.name, linewidth=1)

    axes.set_ylabel('[%]')
    fig.autofmt_xdate(bottom=0.2, rotation=30, ha='right')
    axes.legend(fontsize=4)
    axes.grid()
    plt.show()

    # plot fees and tax
    interval = np.int(len(portfolio.traders[0].date_history) / 10)
    fig, axes = plt.subplots(nrows=3, ncols=1, sharex=True, dpi=150)
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    axes[0].xaxis.set_major_locator(mdates.DayLocator(interval=interval))
    axes[0].set_title('Buy fee history')

    for i, portfolio in enumerate(portfolios_list):
        axes[0].plot(portfolio.traders[0].date_history, np.cumsum(portfolio.traders[0].buy_fee_history),
                     label=portfolio.name, linewidth=1)

    axes[0].set_ylabel('USD')
    axes[0].legend(fontsize=4)
    axes[0].grid()

    axes[1].set_title('Sell fee history')
    for i, portfolio in enumerate(portfolios_list):
        axes[1].plot(portfolio.traders[0].date_history, np.cumsum(portfolio.traders[0].sell_fee_history),
                     label=portfolio.name, linewidth=1)

    axes[1].set_ylabel('USD')
    axes[1].grid()

    axes[2].set_title('Tax history')
    for i, portfolio in enumerate(portfolios_list):
        axes[2].plot(portfolio.traders[0].date_history, np.cumsum(portfolio.traders[0].tax_history),
                     label=portfolio.name, linewidth=1)

    axes[2].set_ylabel('USD')
    axes[2].set_xlabel('Operations')
    axes[2].grid()
    fig.autofmt_xdate(bottom=0.2, rotation=30, ha='right')
    plt.show()



