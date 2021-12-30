import numpy as np
import matplotlib.pyplot as plt
from traders import Trader
from markets import Market
import matplotlib.dates as mdates


def market_plot(market, prm='Open', tickers=None, normalize=True):
    data = market.stocks_data
    if tickers is None:
        tickers = list(data.keys())

    fig = plt.figure(dpi=150)
    ax = fig.add_subplot()
    ax.set_title('Market')
    for ticker in tickers:
        if normalize:
            ax.plot(data[ticker][prm] / data[ticker][prm].min(), label=ticker)
        else:
            ax.plot(data[ticker][prm], label=ticker)
    ax.legend()
    ax.set_xlabel('Date')
    ax.tick_params(axis='x', rotation=70)
    if normalize:
        ax.set_ylabel('Normalized Value')
    else:
        ax.set_ylabel('USD')
    fig.autofmt_xdate(bottom=0.2, rotation=30, ha='right')
    ax.grid()
    plt.show()


def profit_and_portfolio_value(traders: list, parameter: list, parameter_name: str):
    interval = np.int(len(traders[0].date_history) / 10)
    fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True, dpi=150)
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


def profits(traders: list, parameter: list, parameter_name: str):
    interval = np.int(len(traders[0].date_history) / 10)
    fig, axes = plt.subplots(nrows=1, ncols=1, sharex=True, dpi=150)
    axes.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    axes.xaxis.set_major_locator(mdates.DayLocator(interval=interval))
    axes.set_title('profit history')

    for i, trader in enumerate(traders):
        axes.plot(trader.date_history, trader.profit_history, label=parameter_name + ': ' + str(parameter[i]))

    axes.set_ylabel('USD')
    fig.autofmt_xdate(bottom=0.2, rotation=30, ha='right')
    axes.legend()
    axes.grid()
    plt.show()


def portfolio_values(traders: list, parameter: list, parameter_name: str):
    interval = np.int(len(traders[0].date_history) / 10)
    fig, axes = plt.subplots(nrows=1, ncols=1, sharex=True, dpi=150)
    axes.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    axes.xaxis.set_major_locator(mdates.DayLocator(interval=interval))
    axes.set_title('Portfolio value history')

    for i, trader in enumerate(traders):
        axes.plot(trader.date_history, trader.portfolio_value_history, label=parameter_name + ': ' + str(parameter[i]))

    axes.set_ylabel('USD')
    fig.autofmt_xdate(bottom=0.2, rotation=30, ha='right')
    axes.legend()
    axes.grid()
    plt.show()


def liquids(traders: list, parameter: list, parameter_name: str):
    interval = np.int(len(traders[0].date_history) / 10)
    fig, axes = plt.subplots(nrows=1, ncols=1, sharex=True, dpi=150)
    axes.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    axes.xaxis.set_major_locator(mdates.DayLocator(interval=interval))
    axes.set_title('Liquid history')

    for i, trader in enumerate(traders):
        axes.plot(trader.date_history, trader.liquid_history, label=parameter_name + ': ' + str(parameter[i]))

    axes.set_ylabel('USD')
    fig.autofmt_xdate(bottom=0.2, rotation=30, ha='right')
    axes.legend()
    axes.grid()
    plt.show()


def fees_and_tax(traders: list, parameter: list, parameter_name: str):
    interval = np.int(len(traders[0].date_history) / 10)
    fig, axes = plt.subplots(nrows=3, ncols=1, sharex=True, dpi=150)
    axes.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    axes.xaxis.set_major_locator(mdates.DayLocator(interval=interval))
    axes[0].set_title('Buy fee history')

    for i, trader in enumerate(traders):
        axes[0].plot(trader.date_history, np.cumsum(trader.buy_fee_history),
                     label=parameter_name + ': ' + str(parameter[i]))

    axes[0].set_ylabel('USD')
    axes[0].legend()
    axes[0].grid()

    axes[1].set_title('Sell fee history')
    for i, trader in enumerate(traders):
        axes[1].plot(trader.date_history, np.cumsum(trader.sell_fee_history),
                     label=parameter_name + ': ' + str(parameter[i]))

    axes[1].set_ylabel('USD')
    axes[1].legend()
    axes[1].grid()

    axes[2].set_title('Tax history')
    for i, trader in enumerate(traders):
        axes[2].plot(trader.date_history, np.cumsum(trader.tax_history),
                     label=parameter_name + ': ' + str(parameter[i]))

    axes[2].set_ylabel('USD')
    axes[2].set_xlabel('Operations')
    axes[2].legend()
    axes[2].grid()
    plt.show()


def yields(traders: list, parameter: list, parameter_name: str, market: Market):
    interval = np.int(len(traders[0].date_history) / 10)
    fig, axes = plt.subplots(nrows=1, ncols=1, sharex=True, dpi=150)
    axes.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    axes.xaxis.set_major_locator(mdates.DayLocator(interval=interval))
    axes.set_title('Yield history')

    for i, trader in enumerate(traders):
        axes.plot(trader.date_history, trader.yield_history, label=parameter_name + ': ' + str(parameter[i]))

    axes.plot(market.index_data.index.to_numpy(), market.index_return_percent, label='S&P 500')
    axes.set_ylabel('[%]')
    fig.autofmt_xdate(bottom=0.2, rotation=30, ha='right')
    axes.legend()
    axes.grid()
    plt.show()


def yields_usd(traders: list, parameter: list, parameter_name: str, market: Market, liquid):
    interval = np.int(len(traders[0].date_history) / 10)
    fig, axes = plt.subplots(nrows=1, ncols=1, sharex=True, dpi=150)
    axes.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    axes.xaxis.set_major_locator(mdates.DayLocator(interval=interval))
    axes.set_title('Yield history')

    for i, trader in enumerate(traders):
        axes.plot(trader.date_history, trader.portfolio_value_history,
                  label=parameter_name + ': ' + str(parameter[i]))

    axes.plot(market.index_data.index.to_numpy(), (market.index_return_percent / 100 + 1) * liquid,
              label='S&P 500')
    axes.set_ylabel('USD')
    fig.autofmt_xdate(bottom=0.2, rotation=30, ha='right')
    axes.legend()
    axes.grid()
    plt.show()

