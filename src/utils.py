import numpy as np
import matplotlib.pyplot as plt
from traders import Trader
import matplotlib.dates as mdates
import datetime as dt


def plot_trader_data(trader: Trader, interval=20):
    tax_history = trader.tax_history
    buy_fee_history = trader.buy_fee_history
    sell_fee_history = trader.sell_fee_history
    profit_history = trader.profit_history
    liquid_history = trader.liquid_history
    portfolio_value_history = trader.portfolio_value_history
    std_history = trader.std_history
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


def compare_traders(traders: list, parameter: list, parameter_name: str, interval=20):

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
