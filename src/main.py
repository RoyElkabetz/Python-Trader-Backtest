import numpy as np
import argparse
from markets import Market
from brokers import Broker
from traders import Trader
from utils import plot_trader, compare_traders, plot_market
import copy as cp


def simulate(liquid, tickers, periods, ratios, sell_strategy, start_date, end_date, buy_fee,
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

    # plot results
    plot_market(market, normalize=plots_normalize)
    compare_traders(traders_list, periods, 'bp', interval=np.int(len(trader.date_history) / 10))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Backtesting Trading Simulator')

    # Arguments
    parser.add_argument('-tickers', type=str, required=True, nargs='+')
    parser.add_argument('-periods', type=int, required=True, nargs='+')
    parser.add_argument('-ratios', type=float, required=True, nargs='+')
    parser.add_argument('-start_date', type=tuple, default=(2019, 1, 1))
    parser.add_argument('-end_date', type=tuple, default=(2022, 1, 1))
    parser.add_argument('-buy_fee', type=float, default=0.08)
    parser.add_argument('-min_buy_fee', type=float, default=2.)
    parser.add_argument('-sell_fee', type=float, default=0.08)
    parser.add_argument('-min_sell_fee', type=float, default=2.)
    parser.add_argument('-tax', type=float, default=25.)
    parser.add_argument('-liquid', type=float, required=True)
    parser.add_argument('-verbose', type=float, default=True)
    parser.add_argument('-plots_normalize', type=bool, default=True)
    parser.add_argument('-sell_strategy', type=str, default='FIFO', choices=['FIFO', 'LIFO', 'TAX_OPT'])
    args = parser.parse_args()

    # run the simulation
    simulate(args.liquid, args.tickers, args.periods, args.ratios, args.sell_strategy, args.start_date, args.end_date,
             args.buy_fee, args.min_buy_fee, args.sell_fee, args.min_sell_fee, args.tax, args.verbose,
             args.plots_normalize)



