import datetime

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


def draw_figure(canvas, figure):
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=1)
    return figure_canvas_agg


def delete_figure_agg(figure_agg):
    figure_agg.get_tk_widget().forget()
    plt.close('all')


# def traders_plot(dates, data_list):
#
#     fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(6, 6), sharey=True)
#     axes[0, 0].boxplot(data, labels=labels)
#     axes[0, 0].set_title('Default', fontsize=fs)
#
#     axes[0, 1].boxplot(data, labels=labels, showmeans=True)
#     axes[0, 1].set_title('showmeans=True', fontsize=fs)
#
#     axes[0, 2].boxplot(data, labels=labels, showmeans=True, meanline=True)
#     axes[0, 2].set_title('showmeans=True,\nmeanline=True', fontsize=fs)
#
#     axes[1, 0].boxplot(data, labels=labels, showbox=False, showcaps=False)
#     tufte_title = 'Tufte Style \n(showbox=False,\nshowcaps=False)'
#     axes[1, 0].set_title(tufte_title, fontsize=fs)
#
#     axes[1, 1].boxplot(data, labels=labels, notch=True, bootstrap=10000)
#     axes[1, 1].set_title('notch=True,\nbootstrap=10000', fontsize=fs)
#
#     axes[1, 2].boxplot(data, labels=labels, showfliers=False)
#     axes[1, 2].set_title('showfliers=False', fontsize=fs)
#
#     for ax in axes.flatten():
#         ax.set_yscale('log')
#         ax.set_yticklabels([])
#
#     fig.subplots_adjust(hspace=0.4)
#     return fig


def market_plot(market, prm='Open', tickers=None, normalize=True):
    data = market.stocks_data
    if tickers is None:
        tickers = list(data.keys())

    fig = plt.figure()
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
    ax.grid()

    return fig


def compare_traders(traders: list, parameter: list, parameter_name: str, interval=20):

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
    axes[1].legend()
    axes[1].grid()
    return fig


def test():
    import numpy as np
    s = datetime.date(2020, 1, 1)
    d = datetime.timedelta(days=1)
    dates = np.array([s, s+d, s+d+d, s+d+d+d, s+d+d+d+d, s+d+d+d+d+d+d, s+d+d+d+d+d+d+d+d])
    traders = [(i + 1) * np.arange(len(dates)) for i in range(4)]
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 6))
    ax[0, 0].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax[0, 0].xaxis.set_major_locator(mdates.DayLocator(interval=1))
    ax[0, 0].set_title('profit history')

    for i, trader in enumerate(traders):
        ax[0, 0].plot(dates, trader, label=str(i))

    ax[0, 0].set_ylabel('USD')
    ax[0, 0].legend()
    ax[0, 0].grid()
    plt.show()


if __name__ == '__main__':
    test()
