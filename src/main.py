import numpy as np
from markets import Market
from brokers import Broker
from traders import Trader
from utils import plot_trader, compare_traders
import copy as cp

periods = [1, 2, 4]
the_traders = []

tickers = ['AAPL', 'GOOG', 'SPY', 'TSLA', 'ORCL']
market = Market(tickers, start_date=(2019, 1, 1), end_date=(2022, 1, 1))
broker = Broker(buy_fee=0.08, min_buy_fee=2, sell_fee=0.08, min_sell_fee=2, tax=25, my_market=market)
first_date = cp.copy(market.current_date)

for i, period in enumerate(periods):
    print(f'period: {period}')

    # init market
    market.current_idx = 0
    market.current_date = first_date

    # init new trader
    trader = Trader(liquid=50000, balance_period=period, broker=broker, market=market, verbose=True)

    # buy some stocks
    trader.buy('AAPL', 20)
    trader.buy('Goog', 10)
    trader.buy('Tsla', 60)
    trader.buy('ORCL', 40)

    trader_tickers = list(trader.portfolio.keys())
    trader.balance(trader_tickers)

    done = False
    steps = 0

    while not done:
        steps += 1
        done, previous_date = market.step()
        trader.step(previous_date)
        if steps % trader.balance_period == 0:

            trader.balance(trader_tickers)
    the_traders.append(trader)

compare_traders(the_traders, periods, 'bp', interval=np.int(len(trader.date_history) / 10))
