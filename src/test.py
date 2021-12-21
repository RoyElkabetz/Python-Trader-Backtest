from markets import Market
from brokers import Broker
from traders import Trader
import matplotlib.pyplot as plt
from utils import plot_trader, compare_traders, plot_market, profit_and_portfolio_value, compare_fees_and_tax
import copy as cp


def simulator(liquid, tickers, periods, ratios, sell_strategy, start_date, end_date, buy_fee,
              min_buy_fee, sell_fee, min_sell_fee, tax, verbose, plots_normalize):

    traders_list = []
    market = Market(tickers, start_date=start_date, end_date=end_date)
    broker = Broker(buy_fee=buy_fee, min_buy_fee=min_buy_fee, sell_fee=sell_fee,
                    min_sell_fee=min_sell_fee, tax=tax, my_market=market)
    first_date = cp.copy(market.current_date)

    for i, period in enumerate(periods):
        print(f'period: {period}')

        # init market
        market.current_idx = 0
        market.current_date = first_date

        # init new trader
        trader = Trader(liquid=liquid, balance_period=period, broker=broker, market=market,
                        verbose=verbose, sell_strategy=sell_strategy)

        # buy some stocks
        for ticker in tickers:
            trader.buy(ticker, 1)

        done = False
        steps = 0

        trader.balance(tickers, p=ratios)
        while not done:
            steps += 1
            if steps % 100 == 0:
                print('| Step: {:6.0f} / {:6.0f} | Balance period: {:4.0f} |'
                      .format(steps, market.steps, trader.balance_period))
            # step market forward in time
            done, previous_date = market.step()

            # step trader forward in time
            trader.step(previous_date)

            # balance trader portfolio
            if steps % trader.balance_period == 0:
                trader.balance(tickers, p=ratios)

        traders_list.append(trader)

    # # plot market
    # plot_market(market, normalize=plots_normalize)
    return traders_list[0]


if __name__ == '__main__':

    # compare FIFO, LIFO, and Tax minimization strategy
    traders_list = []
    prm = ['FIFO', 'LIFO', 'TAX_OPT']

    # run the simulation
    for strategy in prm:
        print(f'strategy: {strategy}')
        trader = simulator(100000., ['AAPL', 'GOOG', 'SPY', 'ORCL'], [10], [0.25] * 4, strategy,
                           (2018, 1, 1), (2021, 4, 20), 0.08, 2., 0.08, 2., 25., False, True)
        traders_list.append(trader)

    # plot results
    compare_fees_and_tax(traders_list, prm, 'strategy')
    profit_and_portfolio_value(traders_list, prm, 'strategy')


