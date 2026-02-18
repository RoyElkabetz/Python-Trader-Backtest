import argparse
from .markets import Market
from .brokers import Broker
from .traders import Trader
from .utils import market_plot, profit_and_portfolio_value, liquids, fees_and_tax, yields, yields_usd
import copy as cp


def simulator(liquid, tickers, periods, ratios, sell_strategy, start_date, end_date, buy_fee,
              min_buy_fee, sell_fee, min_sell_fee, tax, verbose, plots_normalize,
              deposit, deposit_period, show_plots=True, return_traders=False):

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

            # deposit periodically
            if deposit > 0 and steps % deposit_period == 0:
                trader.deposit(deposit)

            # balance trader portfolio
            if steps % trader.balance_period == 0:
                trader.balance(tickers, p=ratios)

        traders_list.append(trader)

    # plot results
    if show_plots:
        market_plot(market, normalize=plots_normalize)
        profit_and_portfolio_value(traders_list, periods, 'balance period')
        fees_and_tax(traders_list, periods, 'balance period')
        liquids(traders_list, periods, 'balance period')
        yields(traders_list, periods, 'balance period', market)
        yields_usd(traders_list, periods, 'balance period', market, liquid)

    if return_traders:
        return traders_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Backtesting Trading Simulator')

    # Arguments
    parser.add_argument('-tickers', type=str, required=True, nargs='+')
    parser.add_argument('-periods', type=int, required=True, nargs='+')
    parser.add_argument('-ratios', type=float, required=True, nargs='+')
    parser.add_argument('-start_date', type=int, required=True, nargs='+')
    parser.add_argument('-end_date', type=int, required=True, nargs='+')
    parser.add_argument('-deposit', type=float, default=0.0)
    parser.add_argument('-deposit_period', type=int, default=30)
    parser.add_argument('-buy_fee', type=float, default=0.08)
    parser.add_argument('-min_buy_fee', type=float, default=2.)
    parser.add_argument('-sell_fee', type=float, default=0.08)
    parser.add_argument('-min_sell_fee', type=float, default=2.)
    parser.add_argument('-tax', type=float, default=25.)
    parser.add_argument('-liquid', type=float, required=True)
    parser.add_argument('-verbose', type=bool, default=True)
    parser.add_argument('-plots_normalize', type=bool, default=True)
    parser.add_argument('-show_plots', type=bool, default=True)
    parser.add_argument('-return_traders', type=bool, default=False)
    parser.add_argument('-sell_strategy', type=str, default='FIFO', choices=['FIFO', 'LIFO', 'TAX_OPT'])
    args = parser.parse_args()

    # run the simulation
    simulator(args.liquid, args.tickers, args.periods, args.ratios, args.sell_strategy, args.start_date, args.end_date,
              args.buy_fee, args.min_buy_fee, args.sell_fee, args.min_sell_fee, args.tax, args.verbose,
              args.plots_normalize, args.deposit, args.deposit_period, args.show_plots, args.return_traders)
