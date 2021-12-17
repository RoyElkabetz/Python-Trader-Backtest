import numpy as np
from markets import Market
from brokers import Broker
from traders import Trader
from utils import plot_trader, compare_traders, plot_market
import copy as cp

periods = [8]
the_traders = []

tickers = ['AAPL', 'GOOG', 'SPY', 'TSLA', 'ORCL']
p = [0.1, 0.03, 0.07, 0.8]
market = Market(tickers, start_date=(2019, 1, 1), end_date=(2022, 1, 1))
broker = Broker(buy_fee=5., min_buy_fee=2, sell_fee=0.08, min_sell_fee=2, tax=25, my_market=market)
first_date = cp.copy(market.current_date)

for i, period in enumerate(periods):
    print(f'period: {period}')

    # init market
    market.current_idx = 0
    market.current_date = first_date

    # init new trader
    trader = Trader(liquid=50000, balance_period=period, broker=broker, market=market, verbose=False)

    # buy some stocks
    trader.buy('AAPL', 20)
    trader.buy('Tsla', 60)
    trader.buy('ORCL', 40)
    trader.buy('Goog', 10)

    trader_tickers = list(trader.portfolio.keys())
    done = False
    steps = 0

    trader.balance(trader_tickers, p=p)
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
            trader.balance(trader_tickers, p=p)

    the_traders.append(trader)

# plot results
plot_market(market, normalize=False)
compare_traders(the_traders, periods, 'bp', interval=np.int(len(trader.date_history) / 10))
