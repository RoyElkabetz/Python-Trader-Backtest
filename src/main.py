import numpy as np
import matplotlib.pyplot as plt
from markets import Market
from brokers import Broker
from traders import Trader

# periods = [1, 2, 4, 8, 16, 32, 64]
periods = [10]
fees = [0.05]
profit_data = []

for i, period in enumerate(periods):
    print(period)
    # init players
    tickers = ['AAPL', 'GOOG', 'SPY', 'TSLA', 'ORCL']
    market1 = Market(tickers, start_date=(2019, 1, 1), end_date=(2020, 1, 1))
    broker1 = Broker(buy_fee=fees[i], min_buy_fee=1, sell_fee=fees[i], min_sell_fee=1, tax=0.25, my_market=market1)
    trader1 = Trader(liquid=50000, balance_liquid_lim=2000, balance_period=period, my_broker=broker1, my_market=market1)

    # buy some stocks
    trader1.buy('AAPL', 20)
    trader1.buy('Goog', 10)
    trader1.buy('Tsla', 60)
    trader1.buy('ORCL', 40)

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
for i, fee in enumerate(fees):
    plt.plot(profit_data[i], label=str(fee))
plt.xlabel('Days')
plt.ylabel('USD')
plt.legend()
plt.show()
