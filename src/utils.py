import numpy as np
import matplotlib.pyplot as plt
from traders import Trader
from markets import Market
import matplotlib.dates as mdates


def plot_market(market: Market, prm='Open', tickers=None, normalize=True):
    data = market.stocks_data
    if tickers is None:
        tickers = list(data.keys())

    fig = plt.figure(dpi=200)
    ax = fig.add_subplot()
    ax.set_title('Market')
    for ticker in tickers:
        if normalize:
            ax.plot(data[ticker][prm] / data[ticker][prm].min(), label=ticker)
        else:
            ax.plot(data[ticker][prm], label=ticker)
    ax.legend()
    ax.set_xlabel('Date')
    if normalize:
        ax.set_ylabel('Normalized Value')
    else:
        ax.set_ylabel('USD')
    fig.autofmt_xdate(bottom=0.2, rotation=30, ha='right')
    ax.grid()
    plt.show()


def plot_index(market: Market):
    index_return_percent = market.index_return_percent

    interval = np.int(len(market.index_return_percent) / 10)
    fig = plt.figure(dpi=200)
    ax = fig.add_subplot()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=interval))
    ax.set_title('Market Index')
    ax.plot(market.index_data.index.to_numpy(), index_return_percent, label='S&P 500')
    ax.legend()
    ax.set_xlabel('Date')
    ax.set_ylabel('USD')
    fig.autofmt_xdate(bottom=0.2, rotation=30, ha='right')
    ax.grid()
    plt.show()


def plot_trader(trader: Trader, interval=20):
    tax_history = trader.tax_history
    buy_fee_history = trader.buy_fee_history
    sell_fee_history = trader.sell_fee_history
    profit_history = trader.profit_history
    liquid_history = trader.liquid_history
    portfolio_value_history = trader.portfolio_value_history
    std_history = trader.error_history
    date_history = trader.date_history

    # plots
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=interval))
    plt.title('profit history')
    plt.plot(date_history, profit_history)
    plt.ylabel('USD')
    plt.gcf().autofmt_xdate()
    plt.show()

    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=interval))
    plt.title('portfolio value history')
    plt.plot(date_history, portfolio_value_history)
    plt.ylabel('USD')
    plt.gcf().autofmt_xdate()
    plt.show()

    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=interval))
    plt.title('liquid history')
    plt.plot(date_history, liquid_history)
    plt.ylabel('USD')
    plt.gcf().autofmt_xdate()
    plt.show()

    plt.figure()
    plt.title('Tax and Fees')
    plt.plot(np.arange(len(tax_history)), tax_history, label='Tax')
    plt.plot(np.arange(len(buy_fee_history)), buy_fee_history, label='Buy fee')
    plt.plot(np.arange(len(sell_fee_history)), sell_fee_history, label='Sell fee')
    plt.xlabel('a.u.')
    plt.ylabel('USD')
    plt.legend()
    plt.show()

    plt.figure()
    plt.title('portfolio std')
    plt.plot(np.arange(len(std_history)), std_history)
    plt.xlabel('a.u.')
    plt.ylabel('USD')
    plt.legend()
    plt.show()


def compare_traders(traders: list, parameter: list, parameter_name: str):
    interval = np.int(len(traders[0].date_history) / 10)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=interval))
    plt.title('profit history')

    for i, trader in enumerate(traders):
        plt.plot(trader.date_history, trader.profit_history, label=parameter_name + ': ' + str(parameter[i]))

    plt.ylabel('USD')
    plt.gcf().autofmt_xdate()
    plt.legend()
    plt.grid()
    plt.show()

    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=interval))
    plt.title('portfolio volume history')

    for i, trader in enumerate(traders):
        plt.plot(trader.date_history, trader.portfolio_value_history, label=parameter_name + ': ' + str(parameter[i]))

    plt.ylabel('USD')
    plt.gcf().autofmt_xdate()
    plt.legend()
    plt.grid()
    plt.show()


def profit_and_portfolio_value(traders: list, parameter: list, parameter_name: str):
    interval = np.int(len(traders[0].date_history) / 10)
    fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True)
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    axes[0].xaxis.set_major_locator(mdates.DayLocator(interval=interval))
    axes[0].set_title('profit history')

    for i, trader in enumerate(traders):
        axes[0].plot(trader.date_history, trader.profit_history, label=parameter_name + ': ' + str(parameter[i]))

    axes[0].set_ylabel('USD')
    axes[0].legend()
    axes[0].grid()

    axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    axes[1].xaxis.set_major_locator(mdates.DayLocator(interval=interval))
    axes[1].set_title('portfolio volume history')

    for i, trader in enumerate(traders):
        axes[1].plot(trader.date_history, trader.portfolio_value_history, label=parameter_name + ': ' + str(parameter[i]))

    axes[1].set_ylabel('USD')
    fig.autofmt_xdate(bottom=0.2, rotation=30, ha='right')
    axes[1].legend()
    axes[1].grid()
    plt.show()


def compare_fees_and_tax(traders: list, parameter: list, parameter_name: str):
    fig, axes = plt.subplots(nrows=3, ncols=1)
    axes[0].set_title('Tax history')

    for i, trader in enumerate(traders):
        axes[0].plot(trader.tax_history, label=parameter_name + ': ' + str(parameter[i]))

    axes[0].set_ylabel('USD')
    axes[0].legend()
    axes[0].grid()
    axes[1].set_title('Buy fee history')

    for i, trader in enumerate(traders):
        axes[1].plot(trader.buy_fee_history, label=parameter_name + ': ' + str(parameter[i]))

    axes[1].set_ylabel('USD')
    axes[1].legend()
    axes[1].grid()
    axes[2].set_title('Sell fee history')

    for i, trader in enumerate(traders):
        axes[2].plot(trader.sell_fee_history, label=parameter_name + ': ' + str(parameter[i]))

    axes[2].set_ylabel('USD')
    axes[2].legend()
    axes[2].grid()
    plt.show()