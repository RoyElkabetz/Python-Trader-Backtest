import numpy as np
import matplotlib.pyplot as plt
from markets import Market
from brokers import Broker
from traders import Trader

periods = [1, 2, 4, 8, 16, 32, 64]
profit_data = []

for period in periods:
    # init players
    tickers = ['AAPL', 'GOOG', 'SPY', 'TSLA', 'ORCL']
    market1 = Market(tickers, start_date=(2019, 1, 1), end_date=(2020, 1, 1))
    broker1 = Broker(buy_fee=0.2, min_buy_fee=1, sell_fee=0.2, min_sell_fee=1, tax=0.25, my_market=market1)
    trader1 = Trader(liquid=100000, balance_liquid_lim=5000, balance_period=period, my_broker=broker1, my_market=market1)

    # buy some stocks
    trader1.buy('AAPL', 10)
    trader1.buy('AAPL', 10)
    trader1.buy('Goog', 4)
    trader1.buy('Tsla', 6)
    trader1.buy('ORCL', 6)

    trader_tickers = list(trader1.portfolio.keys())
    percentages = [1. / len(trader_tickers)] * len(trader_tickers)
    done = False
    steps = 0

    while not done:
        steps += 1
        done, previous_date = market1.step()
        trader1.step(previous_date)
        if steps % trader1.balance_period == 0:
            trader1.balance(trader_tickers, percentages)

    profit_data.append(trader1.profit_history)


plt.figure()
for i, period in enumerate(periods):
    plt.plot(profit_data[i], label=str(period))
plt.xlabel('Days')
plt.ylabel('USD')
plt.legend()
plt.show()
